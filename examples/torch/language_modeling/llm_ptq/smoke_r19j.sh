#!/usr/bin/env bash
# smoke_r19j.sh — fast end-to-end validation of the R19j-patched AWQ pipeline.
#
# Runs the REAL quantize_quark.py + patched awq.py on a tiny model
# (Qwen3-0.6B) with two knobs forced to their extreme positions:
#   QUARK_AWQ_SANITIZE_CLIP=1.0  -> healthy O(1)-O(10) activations exceed the
#                                   clip, so [AWQ_HARDEN_V3] MUST fire on every
#                                   layer: proves sanitize -> loss -> scale
#                                   search -> export all work on clipped input.
#   QUARK_MAX_MEMORY_GPU_GIB=0.5 -> device map pushes most layers to cpu/meta:
#                                   proves the GPU->CPU boundary crossing (the
#                                   R19h/R19i death zone) under the new env set
#                                   (no expandable_segments, REPLAY_CHUNK=2).
# PPL will be degraded (inputs distorted by design) — this tests MECHANICS,
# not quality. Success = markers fire, boundaries cross, run completes, PPL prints.
set -uo pipefail
# --- path configuration (env-overridable; defaults = R19j box layout) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${QUARK_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
VENV_BIN="${QUARK_VENV_BIN:-$REPO_ROOT/.venv/bin}"
cd "$SCRIPT_DIR" || exit 1
[ -d "$VENV_BIN" ] && export PATH="$VENV_BIN:$PATH"

SMOKE_MODEL="${QUARK_SMOKE_MODEL:-/home/user2/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
SMOKE_OUT="${QUARK_SMOKE_OUT:-/mnt/NVME2/AI/models/_smoke_r19j_out}"
LOG="${SMOKE_LOG:-smoke_r19j.log}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[SMOKE] DRY_RUN=1: resolved config OK:"
  echo "  SMOKE_MODEL=$SMOKE_MODEL"
  echo "  SMOKE_OUT=$SMOKE_OUT"
  echo "  VENV_BIN=$VENV_BIN"
  exit 0
fi

if [ ! -f "$SMOKE_MODEL/config.json" ]; then
  echo "[SMOKE] ERROR: smoke model missing at $SMOKE_MODEL" >&2
  exit 1
fi
rm -rf "$SMOKE_OUT"

echo "[SMOKE] start $(date)"
HIP_VISIBLE_DEVICES=0,1 \
TORCHDYNAMO_DISABLE=1 MIOPEN_FIND_MODE=FAST AMD_SERIALIZE_KERNEL=3 \
HIP_LAUNCH_BLOCKING=1 HSA_ENABLE_SDMA=0 \
QUARK_MAX_MEMORY_GPU_GIB=0.5 QUARK_AWQ_REPLAY_CHUNK=2 \
QUARK_AWQ_SANITIZE_CLIP=1.0 \
QUARK_AWQ_MEMDBG=1 QUARK_AWQ_F32_FALLBACK=1 QUARK_AWQ_MEMDBG_GRID=1 \
QUARK_EXPORT_SYNC=1 \
python3 quantize_quark.py \
  --model_dir "$SMOKE_MODEL" \
  --output_dir "$SMOKE_OUT" \
  --quant_scheme mxfp4 \
  --quant_algo awq \
  --num_calib_data 4 \
  --seq_len 64 \
  --data_type auto \
  --device cuda \
  --multi_gpu balanced \
  --multi_device \
  --model_export hf_format \
  --trust_remote_code \
  --model_attn_implementation sdpa \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[SMOKE] exit=$rc"
echo "--- marker census ---"
grep -c "AWQ_HARDEN_V3" "$LOG" || true
grep -c "dev=meta" "$LOG" || true
grep -E "Perplexity:" "$LOG" | tail -2
if [ "$rc" -eq 0 ]; then echo "[SMOKE] VERDICT: SMOKE_OK"; else echo "[SMOKE] VERDICT: SMOKE_FAIL rc=$rc"; fi