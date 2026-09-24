# 05 — Serving & evaluation

## Serve config of record (custom radiance/dflash vLLM image)

Docker compose (`compose-rendered-docker-ppl.yml` pattern): mount the checkpoint
at `/models:ro`, port 5678, and serve with:

```
--model /models   (positional FIRST — vLLM 0.27's serve CLI requires it;
                   putting --model=... first raises a ValueError in argparse_utils.parse_args)
--enforce-eager
--kv-cache-dtype=fp8
--tensor-parallel-size=2
--attention-backend=R4D
--gpu-memory-utilization=0.75
--max-model-len=204800
--max-num-seqs=8
--max-num-batched-tokens=8192
--enable-prefix-caching
--mamba-cache-mode=align
```

plus env `VLLM_RPC_TIMEOUT=1800` (issue #7). For eval launches, drop speculative
decoding (dflash) to shorten warmup. Swap checkpoints by changing only the
`/models` volume line — keep everything else identical for head-to-heads.

Notes:

- The image runs a battery of patches at container start (quark w8a8/fp8 dynpt,
  dflash base/2/fused-kv/w4/calib/mxfp4-kv, rmsquant fusion, verify head, kv group
  size, topk composite, GDN shared-build/merge-inproj, dynwidth, ar geometry) and
  refreshes radiance modules into site-packages. If you build your own image,
  that patch list IS the compatibility surface — our checkpoint had to match its
  expectations (hence the fp8_mtp conversion, issue #8).
- Clearing the vLLM cache dir between model swaps has been needed; if the server
  misbehaves after a mount swap, suspect stale cache before suspecting the
  checkpoint.

## PPL methodology (quark-exact metric via deployed vLLM)

`ppl_vllm_wikitext.py` replicates quark's `ppl_eval` exactly so numbers are
comparable to quark-reported baselines:

- **Data:** `Salesforce/wikitext` `wikitext-2-raw-v1` test split; text =
  `"\n\n".join(all lines)`; tokenized ONCE with the model's tokenizer.
- **Windows:** NON-overlapping 2048-token chunks (trailing <2048 discarded).
- **Scoring:** causal NLL at every position inside each window (position 0
  unscored), via `/v1/completions` with raw token-id prompts,
  `prompt_logprobs=0`, `max_tokens=1`, `temperature=0`.
- **Metric:** `exp(mean per-position NLL)` over `nsamples × 2047` positions.
  (Quark multiplies per-window mean loss by 2048 then divides by nsamples×2048 —
  the factors cancel.)
- Client hardening: `--batch 2 --sleep 3 --retries 4` (exponential backoff
  5/10/20/40 s capped 60 s) — transient 5xx/connection errors don't kill the run,
  and the gentle cadence avoids overloading a busy server.

Expected shape on this box: ~296,815 scored positions, ~320 s wall-clock. Results
are deterministic across re-runs (±0.0000 same checkpoint; ±~0.01 across
checkpoints/runs).

**Text PPL is vision-tower-independent** — either variant (bf16-vision parity or
vision-quantized) yields the same number. Use the vision probe for vision.

## Quality baselines & head-to-head (this serving stack)

| Model | wikitext-2 PPL (this stack) | Notes |
|---|---|---|
| Huihui BF16 source (HF-eager original) | 6.0167 | pre-quantization reference |
| AMD ref MXFP4 (original measurement env) | 6.5056 | env not recorded — different stack |
| **AMD ref `Qwen3.8-27B-MXFP4-mtpfp8`** | **7.1883** | served under THIS stack |
| orcarouter uncensored MXFP4 (64@512 calib, diff source) | 7.2978 | third-party, same quark recipe |
| **Ours: Huihui MXFP4 W+A (128@512 calib)** | **7.3778** | PRIMARY/MtpFp8 |

Readings:

- **Config cost ≈ +0.68**: even the AMD reference scores 7.1883 here vs its
  original 6.5056. Absolute PPL is only comparable within one serving stack
  (fp8 KV + R4D attention + eager + no-spec all contribute).
- **Body gap = +0.19** (ours vs AMD under identical config): source-model diff
  (Huihui-abliterated vs AMD base) + calibration + algo version (quark 0.12 vs
  0.13). See open item #10 in [03](03-known-issues-and-fixes.md).
- Fair parity target under this stack ≈ 7.19 + ε.
- Cheapest disambiguating experiment still pending: serve the Huihui BF16 source
  under this stack → pure quantization cost = 7.3778 − M.

## Vision probe

`vision_probe.py`: auto-discovers the served model id from `/v1/models`,
synthesizes an unambiguous test image (red circle TL / blue square TR / green
upward triangle BL / "73" BR) with PIL, POSTs it to `/v1/chat/completions`
multimodal, prints the caption. A healthy ViT names all four elements with
positions; a broken tower hallucinates. Pass `--image /path/to/photo.jpg` for a
real-photo check. Run it after ANY export touching the vision path (and after any
serve-config change that could affect multimodal routing).

## What "done" looks like for a new Quark model

1. Export finished under the watchdog with a clean log ([02](02-export-recipe.md)).
2. Post-export checklist fully green, including structural key-diff and
   `algo_config: null` ([04](04-post-export-checklist.md)).
3. Loads in the target serving stack; first request survives warmup.
4. Text PPL measured and compared **head-to-head under one config** against the
   relevant reference(s) — not against historical absolutes.
5. Vision verified end-to-end (multimodal models).
6. Provenance recorded: recipe, exclusions, environment (incl. torch/ROCm
   versions), verification results, PPL + serving config.
