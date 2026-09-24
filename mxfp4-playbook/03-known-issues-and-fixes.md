# 03 — Known issues & fixes (catalog)

Every bug hit during the R19j project, in rough order of discovery. Format:
**Symptom → Root cause → Fix → Status.** Items marked *recurring* will bite you
again on the next export unless the preventive step is done.

---

## 1. gfx1201 "wedge": stochastic stuck kernel mid-AWQ-loop *(recurring until system fix)*

- **Symptom:** run dies silently — GPU pegged 86–91 W, log frozen, no XCP/timeout.
  Always on the actively-computing cuda:1 layer, always at the last scale-search
  grid step (4/4 data points pre-fix).
- **Root cause:** flaky wake-from-GFXOFF (ROCm#6396, RDNA4 driver hard hang,
  independent of PyTorch). Serialized short kernels create micro-idle windows;
  buggy deep-sleep wake leaves the next launch hung forever.
- **Fix:** complete system fix — udev powercap rule + kernel-level
  `virtual_display` heads on BOTH GPUs + GRUB `ppfeaturemask=0xffffffff` kept
  (details in [01](01-system-setup.md)). Post-fix: zero wedges across a full run,
  including every historical failure layer. Watchdog+retry remains as backstop.
- **Status:** fixed (system-level); validated in-situ, not just by stress test.

## 2. Export crash: GDN `_cpu_norm` tied weights *(one-time, code-fixed)*

- **Symptom:** `save_pretrained` dies with "un-declared tied weights" — two
  state-dict keys sharing one `data_ptr`.
- **Root cause:** transformers' `Qwen3_5GatedDeltaNet.forward` lazily builds
  `self._cpu_norm` (an alias of `norm.weight`) when a layer takes the CPU branch
  (ours: layers 61/62, which were CPU-spilled). The alias leaks into the state
  dict; only `lm_head` is a declared tie.
- **Fix:** `_strip_gdn_cpu_norm_artifacts(model)` before export in
  `quantize_quark.py`, plus the `--resume_guard` fast path so the fix doesn't cost
  another 11-h calibration.
- **Status:** fixed in driver. Any future hybrid GDN model exported with some
  layers on CPU needs this strip.

## 3. Clip-search OOM (class #7) *(fixed via co-tiling)*

- **Symptom:** OOM during AWQ clip search on large layers.
- **Root cause:** clip search materialized a full-token working set.
- **Fix:** `QUARK_AWQ_CLIP_TOKENS=512 QUARK_AWQ_CLIP_CO_BATCH=32` — AMD-faithful
  token count processed in co-tiled batches.
- **Status:** fixed; keep both vars.

## 4. NaN boundary blowup across quantized layers *(fixed via HARDEN_V3)*

- **Symptom:** a layer's calibration input carried millions of non-finite values +
  finite ~1e17 (residual-stream blowup accumulating across already-quantized
  layers); downstream math overflowed even after clipping inf to ±1e18; async
  illegal-memory-access abort at the first cross-device copy sync (ROCm#6603 +
  expandable_segments near-ceiling regime).
- **Root cause:** loose sanitize clip (V2: 1e18 = 4× max-finite) + NaN reference
  outputs feeding the loss/scale search + allocator corrupt regime.
- **Fix:** HARDEN_V3 — sqrt-safe clip at `1e4` on inputs AND finite outliers
  (`QUARK_AWQ_SANITIZE_CLIP`), `[AWQ_SANITIZE_OUT]` sanitizes the reference output
  after the f32 fallback, `QUARK_AWQ_F32_FALLBACK=1`, and NO
  `expandable_segments`.
- **Status:** fixed; the final run crossed all boundaries cleanly.

## 5. vLLM load crash: `algo_config` list-of-dicts *(recurring for quark-0.12 exports)*

- **Symptom:** custom vLLM image fails at model load:
  `apply_vllm_mapper → ... key.endswith(".kv_scale") → AttributeError: 'dict'
  object has no attribute 'endswith'`.
- **Root cause:** quark 0.12.post1 serializes
  `quantization_config.algo_config` as a **list containing one dict** (the AWQ
  recipe). vLLM's HF→vLLM name mapper walks every list-valued quant-config field
  assuming string elements. AMD's checkpoints have `algo_config: null` → skipped.
- **Fix:** set `algo_config: null` in `config.json` post-export (backup the file
  first). Verify zero remaining non-string list fields in `quantization_config`.
- **Status:** fixed on all R19j variants; do it on every quark-0.12 export
  destined for vLLM. Long-term: normalize in the export/conversion scripts.

## 6. Dropped vision-tower biases → hallucinated vision *(recurring for the parity-restore path; now code-fixed)*

- **Symptom:** deployed model described images plausibly but wrong — severe
  hallucination — while text PPL was at parity and other MXFP4 models had perfect
  vision.
- **Root cause:** the vision-parity restore re-injected excluded `model.visual.*`
  linears as bf16 `nn.Linear(..., bias=False)` and copied only `.weight` from the
  source. Exactly 110 bias tensors vanished (27 × {attn.qkv, attn.proj,
  mlp.linear_fc1, mlp.linear_fc2} + merger.linear_fc1/fc2). All weights intact ⇒
  text unaffected; every vision projection ran without its learned offset across
  all 27 encoder blocks ⇒ systematically distorted features.
- **Fix (data):** `fix_vision_bias.py` — restores the 110 biases verbatim from the
  BF16 source into the single-shard checkpoint with sha256 verification of every
  pre-existing tensor + source-match of restored biases, rename-backup + atomic
  swap. Idempotent; doubles as a verifier ("OK: all 166 visual biases present").
- **Fix (driver):** `_restore_vision_bf16` now collects matching 1-D bf16 `.bias`
  and builds `Linear(bias=True)` with the source bias attached.
- **Lesson:** a tensor can't vanish from float math — structural bugs like this are
  only caught by a **key-diff against the source checkpoint**. It's in the
  post-export checklist ([04](04-post-export-checklist.md)) for exactly this reason.
- **Status:** fixed on PRIMARY + MtpFp8 (verified bit-exact + live end-to-end via
  multimodal captioning); `-VisionQuant` variant was unaffected (quantized linears
  kept biases through the quark path).

## 7. First-forward RPC timeout kills the engine *(recurring on cold vLLM starts)*

- **Symptom:** first inference request → 500 (`TimeoutError: RPC call to
  sample_tokens timed out`) → EngineDeadError → later requests see connection
  refused. Looks like a bad checkpoint; isn't (weights loaded, prefill scheduled,
  clean RPC timeout).
- **Root cause:** first-forward warmup (~4 min: inductor+aiter JIT, cudagraph
  capture, drafter init) exceeds vLLM's internal RPC deadline.
- **Fix:** `VLLM_RPC_TIMEOUT=1800` env on the server + `--enforce-eager` + no
  speculative decoding for eval launches (shortens warmup).
- **Status:** fixed in serve config; keep the env var.

## 8. Custom vLLM image: MTP module load crash *(fixed via fp8_mtp conversion)*

- **Symptom:** the radiance/dflash vLLM build crashed loading our checkpoint:
  un-named bf16 `mtp.*` weights fell through to global MXFP4 handling → half-width
  packed-param mismatch.
- **Root cause:** that image expects the MTP projections in FP8 with per-channel
  scales (as in AMD's `mtpfp8` reference).
- **Fix:** `fp8_mtp.py` conversion (MXFP4 body + FP8 drafter in one checkpoint):
  8 mtp projections → F8_E4M3 + F32 scales (rel err 2.2–2.6%, vs ~11.6% for
  MXFP4), norms stay BF16, `exclude` + `layer_quant_config` updated accordingly.
  Output structurally identical to the local AMD mtpfp8 reference.
- **Status:** applied → `...-MtpFp8` artifact. Run this (or equivalent) whenever
  targeting that serving stack. Note the conversion must also normalize
  `algo_config → null` so it can't carry issue #5 forward.

## 9. Venv torch / system ROCm version mismatch *(found by audit; fix before next run)*

- **Symptom:** none directly — found by checking versions. Venv ran
  torch 2.13.0a0+rocm7.13.0a (alpha) on a ROCm 10.0 system; ComfyUI's venv had
  the correct +rocm10.0.0 build (different python version).
- **Impact assessment:** limited, because the wheel links system math libs
  (`libhipblas/MIOpen/rccl → /opt/rocm/lib`), so calibration GEMMs used system
  rocBLAS; only the torch frontend was the alpha. Ruled out as the cause of
  issues #1–#8; could contribute a fraction of the +0.19 PPL gap (see #10).
- **Fix:** rebuild the venv with the matching `rocm10.0` cpXY wheel before the
  next PTQ run; verify with `pip show torch` + `ldd libtorch_hip.so`.
- **Status:** documented; rebuild pending (not needed to trust current checkpoints).

## 10. Quality gap: +0.19 PPL vs AMD reference under identical serving *(open, decomposed)*

- **Facts:** under ONE identical serving config (custom vLLM, eager, TP=2, R4D,
  fp8 KV): AMD ref **7.1883** < orcarouter 64@512 **7.2978** < ours 128@512
  **7.3778**. Even the AMD reference misses its original 6.5056 number under this
  stack (≈ +0.68 config cost — the original measurement environment wasn't
  recorded).
- **Decomposition:** the +0.19 splits into (a) source-model diff
  (Huihui-abliterated vs AMD base) + (b) calibration/algo-version cost (quark
  0.12 vs 0.13) + possibly (c) a sliver from the torch mismatch (#9). Evidence
  points at (a) dominating: the third-party same-recipe point sits between the
  others, and cross-build jitter is typically ±0.01–0.05.
- **Cheapest next experiment:** serve the Huihui BF16 source under the identical
  config → pure quantization cost = 7.3778 − M (expect M ≈ 6.7 if the +0.68
  config-cost inference holds). Do NOT re-export solely to chase the torch
  variable before answering the source-model question.
- **Status:** open; numbers stable across re-runs (±0.01).
