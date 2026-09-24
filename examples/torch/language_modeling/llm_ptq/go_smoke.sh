#!/usr/bin/env bash
# One-shot detached launcher: R19j smoke pipeline. Returns immediately.
# Env passthrough: DRY_RUN=1, QUARK_SMOKE_MODEL, QUARK_SMOKE_OUT, ... reach smoke_r19j.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
OUT="${SMOKE_NOHUP_OUT:-nohup_smoke_r19j.out}"
nohup bash smoke_r19j.sh > "$OUT" 2>&1 &
echo "smoke_r19j launched pid=$! log=$OUT"
