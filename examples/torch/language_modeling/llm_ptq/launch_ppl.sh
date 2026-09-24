#!/usr/bin/env bash
# One-shot gentle PPL eval against a deployed vLLM (defaults = R19j box layout).
# Override with: BASE_URL, MODEL, PPL_MODEL_DIR, PPL_LOG.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${QUARK_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
VENV_BIN="${QUARK_VENV_BIN:-$REPO_ROOT/.venv/bin}"
cd "$SCRIPT_DIR" || exit 1
[ -d "$VENV_BIN" ] && export PATH="$VENV_BIN:$PATH"

export BASE_URL="${BASE_URL:-http://localhost:5678}"
export MODEL="${MODEL:-Qwen3.8-27B-Quark-AWQ-MXFP4}"
PPL_MODEL_DIR="${PPL_MODEL_DIR:-/mnt/NVME2/AI/models/Huihui-Qwen3.8-27B-Quark-AWQ-MXFP4-MtpFp8}"
PPL_LOG="${PPL_LOG:-ppl_r19j_mtpfp8.log}"

nohup python3 ppl_vllm_wikitext.py \
  --model_dir "$PPL_MODEL_DIR" \
  --batch 2 --sleep 3 \
  > "$PPL_LOG" 2>&1 &
echo "launched pid=$! base_url=$BASE_URL model=$MODEL log=$PPL_LOG"
