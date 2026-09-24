#!/usr/bin/env bash
# drive_r19j.sh — BOUNDARY-FIX self-driving AWQ MXFP4 export with wedge recovery.
#
# Supersedes drive_r19i.sh after attempt 1 crashed at L56 (dev=meta) — the FIRST
# cpu layer of its device map — exactly like R19h died at ITS first-cpu layer
# (L62). Root cause chain (r19i_attempt1.log):
#   1. L56 calib input carried 8.1M non-finite values + finite ~1e17 (residual
#      stream blowup across quantized layers)
#   2. old HARDEN_V2 clipped inf to +-1e18 (4x max-finite) — too loose
#   3. GDN/attention math overflowed even at 1e18 scale -> 100M NaN refs; f32
#      fallback left 15.6M NaNs; V3 saw 3.4e38 peaks
#   4. async illegal-memory-access aborted at the first cross-device copy sync
#      (ROCm#6603 gfx1201 + expandable_segments near-ceiling regime)
#
# Fixes vs R19i:
#   A. awq.py [AWQ_HARDEN_V3]: fixed sqrt-safe clip (QUARK_AWQ_SANITIZE_CLIP,
#      default 1e4) on inputs AND finite outliers; [AWQ_SANITIZE_OUT] sanitizes
#      the reference output after the f32 fallback so loss/V3 never see NaN.
#   B. NO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -> exits the
#      ROCm#6603 corrupt-regime (we have headroom: peak was 28.28G << 30.42G).
#   C. REPLAY_CHUNK=2 again (was 1): boundary crash did not depend on chunk
#      size (R19h chunk2 reached L62, R19i chunk1 reached L56) -> take the ~40%
#      speed back.
# Kept from R19i: mem cap 23G (shorter cuda:1 span = smaller corruption window),
# HSA_ENABLE_SDMA=0, HIP_LAUNCH_BLOCKING=1, AMD_SERIALIZE_KERNEL=3, F32_FALLBACK,
# LOSS_NORM_V3, MEMDBG grid, EXPORT_SYNC, wedge-watchdog auto-relaunch <=5.
set -uo pipefail
# --- path configuration (env-overridable; defaults = R19j box layout) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${QUARK_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
VENV_BIN="${QUARK_VENV_BIN:-$REPO_ROOT/.venv/bin}"
WORK_DIR="${QUARK_WORK_DIR:-/mnt/NVME2/AI}"          # snapshots/guards land here
cd "$SCRIPT_DIR" || exit 1
[ -d "$VENV_BIN" ] && export PATH="$VENV_BIN:$PATH"

OUT="${QUARK_OUT_MODEL:-/mnt/NVME2/AI/models/Huihui-Qwen3.8-27B-Quark-AWQ-MXFP4}"
SRC="${QUARK_SRC_MODEL:-/mnt/NVME2/AI/models/huihui-ai/Huihui-Qwen3.8-27B-abliterated}"
DRIVE_LOG="${DRIVE_LOG:-drive_r19j.log}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"   # cap total attempts to bound wall-clock (bumped 10->12
                                      # after two consecutive gfx1201 wedges ate attempts 2-3)
POLL="${POLL:-60}"                   # s between watchdog polls
STALL_LIMIT=240    # s of log silence => suspect. Safe at 240 because [AWQ-HB] heartbeats
                   # now advance the log every ~2-5s inside BOTH the scale-search grid loop
                   # and the clip-search loop; a real wedge stops emitting within seconds.
POWER_WEDGE=40     # W; a GPU above this with a silent log = stuck kernel (idle ~9W)
DRAIN_TIMEOUT=150  # s to wait for GPUs to return to idle before relaunch (was 300; drains
                   # rarely finish in 300s anyway and proceed at ~17W -- halve the dead time)

log(){ echo "[DRIVE $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$DRIVE_LOG"; }

gpu_max_power(){
  rocm-smi --showpower 2>/dev/null \
    | awk '/Graphics Package Power/{v=$NF; gsub(/[^0-9.]/,"",v); if(v>m)m=v} END{print m+0}'
}
gpu_idle(){ local p; p=$(gpu_max_power); awk -v p="$p" 'BEGIN{exit !(p<15)}'; }

wait_drain(){
  log "waiting for GPUs to drain (idle <15W)..."
  local i
  for i in $(seq 1 $(( DRAIN_TIMEOUT / 5 ))); do
    if gpu_idle; then log "GPUs drained (max power $(gpu_max_power)W)"; return 0; fi
    sleep 5
  done
  log "WARN: GPUs not fully drained after ${DRAIN_TIMEOUT}s (max power $(gpu_max_power)W); proceeding"
}

kill_run(){
  local pgid="$1"
  log "killing run (pgid=$pgid)"
  kill -TERM -- "-$pgid" 2>/dev/null
  pkill -TERM -f "quantize_quark.py" 2>/dev/null
  sleep 15
  kill -KILL -- "-$pgid" 2>/dev/null
  pkill -KILL -f "quantize_quark.py" 2>/dev/null
  sleep 3
}

log "=== drive_r19j start: BOUNDARY-FIX + CLIP-OOM-FIX (HARDEN_V3 clip 1e4 + SANITIZE_OUT + no expandable_segments, REPLAY_CHUNK=1, CLIP_TOKENS=512 CO_BATCH=32 [AMD-faithful tokens, OOM via co-tiling], mem 26/24 asym), up to ${MAX_ATTEMPTS} attempts ==="
log "config: HIP_LAUNCH_BLOCKING=1 AMD_SERIALIZE_KERNEL=3 NO-expandable_segments QUARK_MAX_MEMORY_GPU_GIB=26 QUARK_MAX_MEMORY_GPU1_GIB=24 REPLAY_CHUNK=1 CLIP_TOKENS=512 CLIP_CO_BATCH=32 SANITIZE_CLIP=1e4"
log "paths: SRC=$SRC OUT=$OUT WORK_DIR=$WORK_DIR VENV_BIN=$VENV_BIN SCRIPT_DIR=$SCRIPT_DIR"
if [ "${DRY_RUN:-0}" = "1" ]; then
  [ -d "$SRC" ] || { log "WARN: SRC model dir not found: $SRC"; }
  log "DRY_RUN=1: resolved config OK, not launching"
  exit 0
fi
# [R19j rebalance 09-06] Both overnight wedges (attempt1 L24, attempt2 L30) fired on
# cuda:1 layers; cuda:0 has processed L0-L23 flawlessly in every run. gfx1201 hard-hang
# (ROCm#6396) bites when cuda:1 is the actively-computing GPU. So: grow gpu0 (26G, proven
# safe, absorbs more mid-range layers -> first-cuda:1-layer moves later) and shrink gpu1
# (14G -> fewer decoder layers compute on the buggy GPU, rest spill to safe CPU GDN path).
# Attacks the confirmed failure site directly; costs a little wall-clock for reliability.
# [R19j cap-up 09-06 ~13:47] After the COMPLETE wedge fix landed (udev powercap + kernel
# virtual_display heads verified connected on BOTH GPUs; gfxoff_stress passed at 24G/GPU),
# gpu1 raised 14->24G: pull the starved mid-range layers back off the slow CPU GDN path.
# gpu0 stays 26G (its transient calibration working-set already runs it to ~27.4G resv).

EARLY_SNAP="$WORK_DIR/quark_post_awq_model_r19j.pt"
EXPORT_GUARD="$WORK_DIR/quark_pre_export_model_r19j.pt"

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  ATLOG="r19j_attempt${attempt}.log"
  rm -rf "$OUT"
  log "=== attempt ${attempt}/${MAX_ATTEMPTS} -> ${ATLOG} ==="

  setsid bash -c '
    HIP_VISIBLE_DEVICES=0,1 \
    TORCHDYNAMO_DISABLE=1 MIOPEN_FIND_MODE=FAST AMD_SERIALIZE_KERNEL=3 \
    HIP_LAUNCH_BLOCKING=1 HSA_ENABLE_SDMA=0 \
    QUARK_MAX_MEMORY_GPU_GIB=26 QUARK_MAX_MEMORY_GPU1_GIB=24 QUARK_AWQ_REPLAY_CHUNK=1 \
    QUARK_AWQ_CLIP_TOKENS=512 QUARK_AWQ_CLIP_CO_BATCH=32 \
    QUARK_AWQ_SANITIZE_CLIP=1e4 \
    QUARK_AWQ_MEMDBG=1 QUARK_AWQ_F32_FALLBACK=1 QUARK_AWQ_MEMDBG_GRID=1 \
    QUARK_EXPORT_SYNC=1 \
    QUARK_EARLY_SNAPSHOT_PATH="'"$EARLY_SNAP"'" \
    QUARK_EXPORT_GUARD_PATH="'"$EXPORT_GUARD"'" \
    exec python3 quantize_quark.py \
      --model_dir "'"$SRC"'" --output_dir "'"$OUT"'" \
      --quant_scheme mxfp4 --quant_algo awq \
      --num_calib_data 128 --seq_len 512 --data_type auto \
      --device cuda --multi_gpu balanced --multi_device \
      --model_export hf_format --trust_remote_code --model_attn_implementation sdpa
  ' > "$ATLOG" 2>&1 &
  PGID=$!
  log "launched pid/pgid=${PGID}"

  armed=0
  last_size=$(stat -c%s "$ATLOG" 2>/dev/null || echo 0)
  last_change=$(date +%s)
  wedged=0
  while kill -0 "$PGID" 2>/dev/null; do
    sleep "$POLL"
    kill -0 "$PGID" 2>/dev/null || break
    size=$(stat -c%s "$ATLOG" 2>/dev/null || echo 0)
    if [ "$size" != "$last_size" ]; then last_size=$size; last_change=$(date +%s); fi
    if [ "$armed" = 0 ] && grep -q "\[MEMDBG\] grid" "$ATLOG" 2>/dev/null; then
      armed=1; last_change=$(date +%s); log "AWQ loop active; arming wedge watchdog"
    fi
    if [ "$armed" = 1 ]; then
      now=$(date +%s); silent=$(( now - last_change )); pw=$(gpu_max_power)
      if [ "$silent" -ge "$STALL_LIMIT" ] && awk -v p="$pw" 'BEGIN{exit !(p>'"$POWER_WEDGE"')}'; then
        log "WEDGE suspected: log silent ${silent}s, gpu power ${pw}W -> killing & relaunching"
        kill_run "$PGID"; wedged=1; break
      fi
    fi
  done

  rc=""
  if ! kill -0 "$PGID" 2>/dev/null; then wait "$PGID"; rc=$?; fi

  if grep -q "Perplexity:" "$ATLOG" 2>/dev/null; then
    cp "$ATLOG" r19j_full.log
    ppl=$(grep -E "Perplexity:" "$ATLOG" | tail -1)
    log "SUCCESS on attempt ${attempt}: ${ppl}"
    log "output dir: $(ls "$OUT" 2>/dev/null | tr '\n' ' ')"
    log "VERDICT: RUN_OK (attempt ${attempt})"
    exit 0
  fi

  if [ "$wedged" = 1 ]; then
    log "attempt ${attempt} ended: WEDGE (killed)"
  else
    log "attempt ${attempt} ended: rc=${rc:-?} (crash/OOM, not a detected wedge)"
  fi
  tail -6 "$ATLOG" 2>/dev/null | sed 's/^/[   tail] /' | tee -a "$DRIVE_LOG"
  wait_drain
  attempt=$(( attempt + 1 ))
done

log "VERDICT: EXHAUSTED ${MAX_ATTEMPTS} attempts without a PPL result"
exit 1