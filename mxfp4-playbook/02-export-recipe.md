# 02 — Export recipe (the actual command + every knob)

## The export command (R19j, verbatim from `drive_r19j.sh`)

```bash
HIP_VISIBLE_DEVICES=0,1 \
TORCHDYNAMO_DISABLE=1 MIOPEN_FIND_MODE=FAST AMD_SERIALIZE_KERNEL=3 \
HIP_LAUNCH_BLOCKING=1 HSA_ENABLE_SDMA=0 \
QUARK_MAX_MEMORY_GPU_GIB=26 QUARK_MAX_MEMORY_GPU1_GIB=24 QUARK_AWQ_REPLAY_CHUNK=1 \
QUARK_AWQ_CLIP_TOKENS=512 QUARK_AWQ_CLIP_CO_BATCH=32 \
QUARK_AWQ_SANITIZE_CLIP=1e4 \
QUARK_AWQ_MEMDBG=1 QUARK_AWQ_F32_FALLBACK=1 QUARK_AWQ_MEMDBG_GRID=1 \
QUARK_EXPORT_SYNC=1 \
QUARK_EARLY_SNAPSHOT_PATH=$QUARK_WORK_DIR/quark_post_awq_model_r19j.pt \
QUARK_EXPORT_GUARD_PATH=$QUARK_WORK_DIR/quark_pre_export_model_r19j.pt \
python3 quantize_quark.py \
  --model_dir <SRC_HF_CHECKPOINT> --output_dir <OUT_DIR> \
  --quant_scheme mxfp4 --quant_algo awq \
  --num_calib_data 128 --seq_len 512 --data_type auto \
  --device cuda --multi_gpu balanced --multi_device \
  --model_export hf_format --trust_remote_code --model_attn_implementation sdpa
```

### Path configuration (env-overridable)

All scripts derive their location from `BASH_SOURCE`, so they run from any
checkout path. Every machine-specific path is an env var whose default is the
R19j box layout — unset them all and behavior on that box is unchanged; set
them for anywhere else:

| Env var | Default | Used by |
|---|---|---|
| `QUARK_REPO_ROOT` | derived (4 levels up from the script dir) | venv discovery in all launchers |
| `QUARK_VENV_BIN` | `$QUARK_REPO_ROOT/.venv/bin` (skipped if absent) | PATH prepend in drive/smoke/ppl launchers |
| `QUARK_SRC_MODEL` | `/mnt/NVME2/AI/models/huihui-ai/Huihui-Qwen3.8-27B-abliterated` | `drive_r19j.sh` (`--model_dir`) |
| `QUARK_OUT_MODEL` | `/mnt/NVME2/AI/models/Huihui-Qwen3.8-27B-Quark-AWQ-MXFP4` | `drive_r19j.sh` (`--output_dir`) |
| `QUARK_WORK_DIR` | `/mnt/NVME2/AI` | snapshot/guard `.pt` locations (drive + driver fallback) |
| `QUARK_SMOKE_MODEL` / `QUARK_SMOKE_OUT` | Qwen3-0.6B HF cache snapshot / `…/models/_smoke_r19j_out` | `smoke_r19j.sh` |
| `BASE_URL` / `MODEL` | `http://localhost:5678` / `Qwen3.8-27B-Quark-AWQ-MXFP4` | `launch_ppl.sh` → `ppl_vllm_wikitext.py` |
| `PPL_MODEL_DIR` / `PPL_LOG` | `…/Huihui-…-MtpFp8` / `ppl_r19j_mtpfp8.log` | `launch_ppl.sh` |
| `DRIVE_LOG` / `SMOKE_LOG` / `SMOKE_NOHUP_OUT` | `drive_r19j.log` / `smoke_r19j.log` / `nohup_smoke_r19j.out` | log locations |
| `MAX_ATTEMPTS` / `POLL` | `12` / `60` s | `drive_r19j.sh` watchdog tuning |
| `QUARK_MODELS_DIR` | `/mnt/NVME2/AI/models` | `fix_vision_bias.py` target/source roots |

**Dry-run mode:** `DRY_RUN=1 bash drive_r19j.sh` (and `smoke_r19j.sh`) resolves
and logs every path, checks the source model dir exists, and exits without
launching — run this first on any new machine to confirm your config resolves
before committing to an ~11-h calibration.

Example fresh-box invocation:

```bash
QUARK_SRC_MODEL=/opt/models/base-27b \
QUARK_OUT_MODEL=/opt/models/out-mxfp4 \
QUARK_WORK_DIR=/opt/quark-work \
DRY_RUN=1 bash drive_r19j.sh        # verify resolution first
# ...then drop DRY_RUN for the real run
```

### Calibration recipe

- **128 samples × 512 tokens, pileval** (`--num_calib_data 128 --seq_len 512`).
  This mirrors the AMD reference recipe. A third-party export at 64×512 scored
  worse (see [05](05-serving-and-evaluation.md) head-to-head). Don't cut samples
  to save time unless you accept a quality cost.
- `--data_type auto` (bf16 source stays bf16 through calibration).

### Layer exclusions

Exclude everything that must stay unquantized:

- `lm_head` (always; also the tied-weight source)
- the full vision tower: prefix `model.visual.` (kept bf16 to match the AMD
  reference; quantizing it costs ~0.67 GB and distorts image features)
- any MTP / speculative module: prefix `mtp.`

In this driver these are handled by the R19j additions rather than plain
`--exclude_layers`: the vision linears are restored to bf16 `nn.Linear` from the
source checkpoint via `_restore_vision_bf16` (now restoring weight **and** bias),
and `_patch_config_exclude_for_bf16_linears` syncs `config.json`'s
`quantization_config.exclude` with what's actually on disk (the exporter otherwise
bakes in only `["lm_head"]`, and config-driven loaders treat `exclude` as the
source of truth for which linears stay unquantized). If you use stock quark CLI
instead, pass `--exclude_layers` with the same prefixes and verify the exported
config matches disk ([04](04-post-export-checklist.md)).

### Env var rationale (why each one exists)

| Var | Value | Why |
|---|---|---|
| `HIP_VISIBLE_DEVICES` | `0,1` | pin both GPUs explicitly |
| `TORCHDYNAMO_DISABLE` | `1` | no dynamo recompiles mid-calibration |
| `MIOPEN_FIND_MODE` | `FAST` | skip exhaustive conv algo search (GDN convs) |
| `AMD_SERIALIZE_KERNEL` | `3` | serialize kernel launches → micro-idle windows (wedge mitigation pair with virtual display); also makes hangs deterministic-ish |
| `HIP_LAUNCH_BLOCKING` | `1` | synchronous launches; same family as above |
| `HSA_ENABLE_SDMA` | `0` | belt-and-suspenders around async copy paths (ROCm#6603 corrupt regime) |
| `QUARK_MAX_MEMORY_GPU_GIB` / `..._GPU1_GIB` | `26` / `24` | accelerate device-map caps below the ~30.42 GiB death-line, minus transient working-set spikes. Asymmetric on purpose: grow the GPU with the flawless record (cuda:0), shrink the buggy-GPU span so fewer layers compute there |
| `QUARK_AWQ_REPLAY_CHUNK` | `1` | smaller per-layer replay working set; boundary crashes turned out not to depend on chunk size, but keep it small near the ceiling |
| `QUARK_AWQ_CLIP_TOKENS` / `QUARK_AWQ_CLIP_CO_BATCH` | `512` / `32` | clip-search co-tiling: AMD-faithful token count without OOMing the clip search (class #7 fix) |
| `QUARK_AWQ_SANITIZE_CLIP` | `1e4` | sqrt-safe clip on inputs AND finite outliers (HARDEN_V3). Old V2 clipped inf to ±1e18 — too loose; GDN/attention math overflowed even at 1e18 |
| `QUARK_AWQ_F32_FALLBACK` | `1` | recompute suspicious layers in fp32 when numerics look wrong |
| `QUARK_AWQ_SANITIZE_OUT` | (on in V3) | sanitize the *reference output* after f32 fallback so loss/scale-search never sees NaN |
| `QUARK_AWQ_MEMDBG` / `..._GRID` | `1` | memory telemetry per grid step; also the log heartbeat the watchdog arms on |
| `QUARK_EXPORT_SYNC` | `1` | synchronous export (cross-device copies can't race) |
| `QUARK_EARLY_SNAPSHOT_PATH` / `QUARK_EXPORT_GUARD_PATH` | cloudpickle `.pt` | pre-export snapshot of the fully-AWQ'd frozen model → enables the `--resume_guard` fast path (below) |

**Do NOT set** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` here — combined
with ROCm#6603 on gfx1201 in the near-ceiling regime it produced an
async illegal-memory-access abort. We had headroom (peak 28.28 G << 30.42 G) so
plain allocation was safe.

## The self-driving loop (watchdog pattern)

Never run the ~11-h export bare. `drive_r19j.sh` wraps it:

- **Wedge detection:** once the AWQ loop is active (log shows `[MEMDBG] grid`),
  arm a watchdog: **≥240 s of log silence AND max GPU power > 40 W** = stuck
  kernel (idle is ~9–17 W). 240 s is safe because `[AWQ-HB]` heartbeats advance
  the log every 2–5 s inside both the scale-search and clip-search loops.
- **Recovery:** TERM→KILL the process group, wait for GPUs to drain (<15 W,
  up to 150 s), relaunch from L0 (no checkpoint/resume in the loop itself).
- **Attempt cap:** 12 total attempts bounds wall-clock. With the complete wedge
  fix the final run finished on attempt 1; pre-fix observed rate was ~50%/run.
- **Success signal:** `Perplexity:` line in the attempt log.

## Pre-flight before the long run

1. System fix verified live (both virtual heads connected, udev cap locked) — [01](01-system-setup.md).
2. `gfxoff_stress.py` passes (~293 s).
3. **Mechanics smoke test** on a tiny model (we used Qwen3-0.6B end-to-end through
   the same driver/env) — catches config/arg/import regressions in minutes instead
   of hours. Launcher pattern: `go_smoke.sh`.
4. Venv torch matches system ROCm ([01](01-system-setup.md)).

## Resume-guard fast path (skip the 11 h on export bugs)

If the AWQ loop completes but the **export** crashes (ours did — GDN `_cpu_norm`
tied-weight bug, see [03](03-known-issues-and-fixes.md) #2), you do NOT want to
redo calibration. The driver snapshots the fully-AWQ'd frozen model to
`QUARK_EXPORT_GUARD_PATH` right before export. After fixing the export code:

```bash
python3 quantize_quark.py \
  --resume_guard /mnt/NVME2/AI/quark_pre_export_model_r19j.pt \
  --restore_vision_bf16 <SRC_HF_CHECKPOINT> \
  --output_dir <OUT_DIR> ...
```

This skips load/calibration/AWQ entirely and goes straight to export. That's how
the final checkpoints were produced in ~1 hour instead of ~24.

## Timing expectations (dual gfx1201, 27B, this recipe)

- AWQ loop: ~750–930 s/layer × 64 layers ≈ **11 h**.
- CPU-spilled layers (over the VRAM cap) are slower still — that's the trade for
  reliability.
- Export + PPL eval after the loop: ~1–2 h.
- Budget 1–2 extra attempts if the wedge fix isn't fully in place.
