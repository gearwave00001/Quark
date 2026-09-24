# MXFP4 AWQ Export Playbook (Qwen3.8-27B-class models, dual gfx1201)

Field guide distilled from the R19j project: producing a max-quality **real-quant
MXFP4 W+A** (weights AND activations FP4, group size 32, E8M0 scales) export of a
~27B multimodal LLM via Quark AWQ PTQ, then serving it in a custom vLLM build and
measuring quality. Everything here was learned the expensive way — read before
committing an ~11-hour calibration run.

## Who this is for

Someone who wants to create their own Quark MXFP4 model from a base HF checkpoint,
on (or similar to) this hardware: 2× AMD Radeon AI PRO R9700 (gfx1201, 32 GB),
ROCm 10.0 userspace, Linux.

## Contents

| File | What it covers |
|---|---|
| [01-system-setup.md](01-system-setup.md) | VRAM death-line, the gfx1201 "wedge" bug and its complete fix (udev + kernel virtual display + GRUB), pre-flight stress test, venv/torch version matching |
| [02-export-recipe.md](02-export-recipe.md) | The exact export command, every env var with rationale, calibration recipe, layer exclusions, the self-driving watchdog loop, smoke-test pre-flight, resume-guard fast path |
| [03-known-issues-and-fixes.md](03-known-issues-and-fixes.md) | Catalog of every bug hit: wedges, GDN `_cpu_norm` export crash, clip-search OOM, NaN boundary blowup, `algo_config` vLLM crash, dropped vision biases, first-forward RPC timeout, MTP-fp8 conversion, torch/ROCm mismatch |
| [04-post-export-checklist.md](04-post-export-checklist.md) | Verification battery to run on every export before you trust or upload it |
| [05-serving-and-evaluation.md](05-serving-and-evaluation.md) | vLLM serve config of record, warmup-timeout fix, PPL methodology (quark-exact metric), quality baselines & head-to-head table, vision probe |
| [site-packages-patches/](site-packages-patches/) | The 8 non-repo code changes (7 in `amd-quark`, 1 in `transformers`) that a fresh checkout is missing — exact diffs, apply script, and a RECORD-hash audit tool to verify any venv |

## TL;DR — the ten things that matter most

1. **Calibration recipe matters more than people expect.** We used AMD's recipe:
   `--num_calib_data 128 --seq_len 512` on pileval. Halving samples (64@512) measurably
   hurt a third-party export (7.2978 vs 7.3778 PPL, different sources aside).
2. **Exclude what shouldn't be quantized, up front:** `lm_head`, the vision tower
   (`model.visual.`), and any speculative/MTP module (`mtp.`). Quantizing the vision
   tower costs ~0.67 GB and distorts features; keeping it bf16 matches the AMD reference.
3. **On gfx1201, install the complete wedge fix BEFORE the long run** (udev powercap
   rule + kernel-level `virtual_display` + GRUB `ppfeaturemask`). Without it, expect
   stochastic ~50%/run driver hangs (ROCm#6396) that burn entire 11-h attempts.
4. **Run the export under a watchdog**, not bare. Log-silence + elevated-GPU-power =
   stuck kernel → kill, drain, relaunch. Cap attempts (we used 12).
5. **VRAM has a practical death-line ~30.42 GiB/GPU** on these cards (not 32). Set
   accelerate device-map caps below it (we ran 26/24 GiB); layers over the cap spill
   to a CPU GDN path that is slow but safe.
6. **Never trust an export blindly.** Run the post-export checklist
   ([04](04-post-export-checklist.md)): tensor-count diff vs source, dtype census,
   config/exclude sync, hash verification. Our worst bug (110 dropped vision biases)
   was invisible to every other check except a key-diff against the source.
7. **Normalize `quantization_config.algo_config` to `null`** after export or vLLM's
   name-mapper crashes on load.
8. **Serve with `VLLM_RPC_TIMEOUT=1800`** — the first-forward cold compile (~4 min)
   otherwise kills the engine mid-warmup.
9. **Match your venv torch to your system ROCm.** We found the venv running a
   ROCm-7.13-alpha torch on a ROCm-10.0 system (it linked the system math libs, so
   damage was limited, but don't rely on that). Audit with `pip show torch` + `ldd`.
10. **Measure PPL the same way every time, under identical serving config.** Absolute
    PPL is only comparable within one serving stack — our stack adds ~+0.68 vs the
    original quark-HF measurement environment. Compare models head-to-head under ONE
    config, not against historical absolute numbers.
11. **A fresh `pip install amd-quark==0.12.post1` will NOT run this recipe.** Eight
    files in the installed packages carry required fixes (qwen3_5 load path,
    offloaded-layer materialization, CPU-GDN fallbacks, sanitize/harden logic).
    Apply [site-packages-patches/](site-packages-patches/) and verify with its audit tool.

## Key artifacts referenced in these docs

All under the Quark repo unless noted:

- `examples/torch/language_modeling/llm_ptq/quantize_quark.py` — the PTQ driver (with R19j additions: `_strip_gdn_cpu_norm_artifacts`, `_restore_vision_bf16`, `_patch_config_exclude_for_bf16_linears`, `--resume_guard`, `--restore_vision_bf16`)
- `examples/torch/language_modeling/llm_ptq/drive_r19j.sh` — self-driving export with wedge watchdog
- `examples/torch/language_modeling/llm_ptq/fix_vision_bias.py` — surgical restorer + post-export verifier for dropped vision biases
- `examples/torch/language_modeling/llm_ptq/ppl_vllm_wikitext.py` — quark-exact wikitext PPL against a deployed vLLM
- `examples/torch/language_modeling/llm_ptq/vision_probe.py` — end-to-end multimodal sanity probe
- `examples/torch/language_modeling/llm_ptq/gfxoff_stress.py` — GFXOFF entry/exit stress pre-flight
- `mxfp4-playbook/site-packages-patches/` — the 8 site-packages diffs + `apply.sh` + `audit_site_packages.py`
- `_r19j-journey/project-log/qwark-mxfp4-project-summary.md` — full chronological project log (every attempt, every decision); the rest of that folder is the preserved journey (logs, superseded scripts, experiments, backups)
