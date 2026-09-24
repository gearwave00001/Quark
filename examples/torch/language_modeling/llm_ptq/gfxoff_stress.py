#!/usr/bin/env python3
"""gfxoff_stress.py — replay the R19j wedge conditions to test the virtual-display/GFXOFF fix.

Our class-#5 wedges: serialized short kernels (HIP_LAUNCH_BLOCKING=1 + AMD_SERIALIZE_KERNEL=3),
GPU near its VRAM ceiling, micro-idle windows between launches => suspected flaky wake-from-GFXOFF
(RDNA4 hard hang, ROCm#6396). This script recreates ALL of those on BOTH GPUs and hammers the
idle->wake transition thousands of times. Clean finish = the wedge class is (likely) gone.
Hang (log stalls + a GPU pegs ~80-90W with no progress) = wedge still present on that GPU.
"""
import torch, time

print("STRESS_START ndev=%d" % torch.cuda.device_count(), flush=True)
t0 = time.time()

# Reserve near-ceiling VRAM on each GPU (6 x 4GB = 24GB) to mirror the resv~27G regime
# at the 26/24 caps (gpu1 cap raised 14->24G on 09-06 after the complete wedge fix).
hold = []
for dev in ("cuda:0", "cuda:1"):
    for _ in range(6):
        hold.append(torch.empty(1_000_000_000, dtype=torch.float32, device=dev))
    print("held ~24G on %s" % dev, flush=True)

a0 = torch.randn(1024, 1024, device="cuda:0"); b0 = torch.randn(1024, 1024, device="cuda:0")
a1 = torch.randn(1024, 1024, device="cuda:1"); b1 = torch.randn(1024, 1024, device="cuda:1")

N = 50000
for i in range(N):
    _ = (a0 @ b0)
    _ = (a1 @ b1)
    if i % 2000 == 0:
        torch.cuda.synchronize()
        print("iter %d/%d t=%.1fs" % (i, N, time.time() - t0), flush=True)
    time.sleep(0.005)  # 5ms micro-idle window each iter -> ~50k GFXOFF entry/exit candidates

torch.cuda.synchronize()
print("STRESS_OK total=%.1fs" % (time.time() - t0), flush=True)