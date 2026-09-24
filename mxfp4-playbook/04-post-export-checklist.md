# 04 — Post-export checklist (run on EVERY export before trusting/uploading)

The R19j post-mortem lesson: the worst bug (110 dropped vision biases, #6 in
[03](03-known-issues-and-fixes.md)) passed every "looks fine" check — size was
right, dtypes were right, text PPL was at parity. Only a **key-diff against the
source** caught it. Run all of these; they're cheap (minutes) relative to an
11-h run.

## 1. Structural diff vs source checkpoint (the one that catches structural bugs)

```bash
python3 fix_vision_bias.py --dry-run   # or extend its key-diff logic to your model
```

What it does / what you want to see:

- Lists every `model.visual.* .bias` present in the source but missing from the
  export. Expect `OK: all N visual biases present (nothing to do)`. If it says
  `restoring 110 ...`, run it for real.
- Generalize for non-vision models: enumerate keys in source vs export both ways
  (`missing = src - out`, `extra = out - src`). Any missing *weight* is a hard
  fail; missing *biases/norms* are the classic silent-damage class.
- Also compare tensor COUNTS per subtree (e.g. our source had 333 `model.visual`
  tensors; the broken exports had 223 — the 110 gap was the fingerprint).

## 2. Dtype census + header sanity

Parse the safetensors header (no tensor loads needed):

- Distinct dtypes are exactly what the recipe implies (ours: `U8` packed MXFP4 +
  `BF16`; the MtpFp8 variant adds `F8_E4M3`/`F32` for the drafter). A surprise
  dtype = wrong path ran.
- File `__metadata__` unchanged (`{'format': 'pt'}`); no per-tensor metadata lost.
- For single-shard checkpoints there's no index.json — confirm which layout you
  actually have before assuming either.

## 3. Value-level spot checks

- Hash (sha256) a sample of restored/copied tensors against the source — expect
  exact match for verbatim-copied bf16 params.
- For quantized linears: count QuantLinears (ours: 496 lm qlinears), confirm
  excluded modules are plain `nn.Linear` with bf16 weight **and bias where the
  source has one**.
- Parity variant: `u8_vision == 0` (no fp4 bytes under `model.visual`).

## 4. config.json audit

- `quantization_config.exclude` matches what's actually on disk (module-name AND
  tensor-name forms where loader matching style is unknown — redundant fnmatch
  patterns are harmless). The exporter bakes in only `["lm_head"]`; the R19j
  `_patch_config_exclude_for_bf16_linears` step syncs the rest. Every bf16
  unquantized tensor on disk must be covered by an exclude pattern, or a
  config-driven loader will try to build a QuantLinear against bf16 weights and
  fail.
- **`algo_config` is `null`** (issue #5). Backups: `config.json.bak_algoconfig`.
- `architectures`, `dtype`, tokenizer/processor files intact. Note: `crc32.txt`
  (if present) guards only tokenizer/config files, not the weights — rewriting
  `model.safetensors` doesn't invalidate it.

## 5. Load test in the target stack

Don't stop at "files look right":

- Load in the actual serving image (vLLM custom build) — this catches mapper and
  packing issues the HF loader never sees.
- First request will need `VLLM_RPC_TIMEOUT=1800` (issue #7) — don't misread the
  cold-start timeout as a checkpoint fault.
- Text smoke: short completion, sane output.
- **Vision smoke (multimodal models):** `vision_probe.py` — synthesize an
  unambiguous image (colored shapes + digits) and require the model to name them.
  This is the end-to-end check that would have caught issue #6 immediately.
- PPL: [05](05-serving-and-evaluation.md).

## 6. Housekeeping

- Keep backups until the load test passes: we used
  `model.safetensors.bak_novisionbias` / `config.json.bak_algoconfig` alongside.
- Record the verification results next to the checkpoint (our log:
  `_r19j-journey/project-log/qwark-mxfp4-project-summary.md`) — sizes, tensor
  counts, PPL, config deltas. Future-you (or the person uploading) needs provenance.
- If uploading to HuggingFace: state base model, method/scheme/group/scales,
  calibration recipe, exclusions, environment, and PPL **with the serving config
  it was measured under** (absolute PPL is stack-dependent — see #10 in
  [03](03-known-issues-and-fixes.md)).
