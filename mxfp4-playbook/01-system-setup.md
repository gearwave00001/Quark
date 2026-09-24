# 01 — System setup (gfx1201 dual-GPU)

## Hardware & the VRAM death-line

- 2× AMD Radeon AI PRO R9700, gfx1201, 32 GB GDDR6 each.
- Kernel `mem_info_vram_total` = **31.86 GiB** per card (full BAR-visible size; MEM ECC
  being active does NOT shrink this — GDDR6 integrates parity into the memory
  architecture rather than carving a separate block).
- Practical torch ceiling: **~30.42 GiB**. The ~1.4 GiB gap is driver/page-table/
  allocator overhead (+ expandable_segments when used). Treat 30.42 as the hard
  line: allocations past it OOM or (worse) enter corrupt-regime territory.
- Keep MEM ECC on (it protects us); GECC (RAS error reporting) off is fine for a
  one-shot export. Neither needs a reboot to keep current config.

Plan your accelerate device-map caps **below** the death-line with headroom for the
transient calibration working set (it spikes above steady-state). We ran
`QUARK_MAX_MEMORY_GPU_GIB=26` / `QUARK_MAX_MEMORY_GPU1_GIB=24`; observed reserved
peaks were ~29.55 G / ~30.07 G respectively. Layers that don't fit spill to a CPU
GDN path — slow but safe.

## The "wedge" bug (ROCm#6396) and its complete fix

### Symptom

A kernel hangs mid-run: GPU pegged at 86–91 W doing nothing, log silent, no XCP,
no timeout — the run is dead until killed. On our box it fired stochastically
(~50% of runs pre-fix), always on the **actively-computing cuda:1 layer**, during
the AWQ scale-search grid loop (which serializes thousands of short kernels).

### Root cause

Flaky **wake-from-GFXOFF** on RDNA4. The GRUB mask
`amdgpu.ppfeaturemask=0xffffffff` unlocks GFXOFF power-gating (compute blocks shut
down immediately when idle). Serialized short kernels create micro-idle windows
between launches → deep-sleep entry/exit → buggy wake leaves the next launch hung
forever. A sibling machine hit the same failure family headless
(`hipModuleUnload failed: unspecified launch failure`).

### The complete fix (all three parts required)

1. **udev powercap rule** — `/etc/udev/rules.d/99-amd-powercap.rules`:
   ```
   ACTION=="add", SUBSYSTEM=="hwmon", ATTRS{device/vendor}=="0x1002", ATTRS{device/device}=="0x7551", ATTR{name}=="amdgpu", ATTR{power1_cap}="215000000"
   ```
   Forces the kernel to process/lock hwmon properties at device-init. Without it the
   box takes a generic init path and the virtual display goes unparsed in text mode.
   Verify: `power1_cap=215000000` on both amdgpu hwmon nodes.

2. **Kernel-level virtual display heads** — `/etc/modprobe.d/amdgpu.conf`:
   ```
   options amdgpu virtual_display=0000:0c:00.0,1;0000:0f:00.0,1
   ```
   (substitute your two GPUs' PCI addresses). No X server needed. Verify:
   `card0-Virtual-1` and `card1-Virtual-2` both `status=connected` — **both** GPUs
   must be covered, since the hang fires on whichever GPU is computing.

3. **GRUB mask kept intact:** `amdgpu.ppfeaturemask=0xffffffff` (LACT preserved).

Reboot after changes. This is a *driver-mitigation*, not a cure — keep the watchdog
([02](02-export-recipe.md)) as backstop.

### Pre-flight stress test

Before committing an 11-h run, exercise the exact failure path:
`gfxoff_stress.py` (in the llm_ptq dir) — 50k idle→wake transitions holding
20–24 G/GPU, serialized kernels (`HIP_LAUNCH_BLOCKING=1`, `AMD_SERIALIZE_KERNEL=3`,
`HSA_ENABLE_SDMA=0`). Pass ≈ 293 s, clean teardown, no KFD residue. Stress passes
are necessary but not sufficient — the real validation is crossing the historical
wedge layers in-situ (first cuda:1 layer, last cuda:1→CPU boundary).

## Venv / torch version matching

Audit finding (2026-09-10): the Quark venv was running **torch 2.13.0a0+rocm7.13.0a**
(an April-2026 alpha) on a **ROCm 10.0** system. It worked because the wheel links
its math libraries against the system (`ldd libtorch_hip.so` shows
`libhipblas/libMIOpen/librccl → /opt/rocm/lib`), so GEMMs ran on system rocBLAS —
but the frontend (op scheduling, reductions, triton codegen) was the mismatched
alpha. Don't rely on that luck.

Check before any run:

```bash
pip show torch            # want +rocm<system-version>, e.g. +rocm10.0.0
/opt/rocm/bin/hipcc --version; dpkg -l | grep amdrocm-core   # system ROCm version
python -c "import torch; print(torch.version.hip, torch.cuda.get_arch_list())"
ldd $(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib/libtorch_hip.so | grep /opt/rocm
```

If mismatched, rebuild the venv with the matching `rocm<sys>` cpXY wheel. Note the
venv's python version constrains which wheels exist (ours was py3.12; the
system-matched build we found was py3.14 in another venv).
