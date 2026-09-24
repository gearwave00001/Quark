#!/usr/bin/env python3
"""fix_vision_bias.py — restore the dropped model.visual.* .bias tensors into the
R19j MXFP4 single-shard exports.

Diagnosis
---------
The export pipeline silently dropped exactly 110 vision-tower bias tensors from
every variant (confirmed by key-diff vs the BF16 source):

    27 x { attn.qkv.bias, attn.proj.bias, mlp.linear_fc1.bias, mlp.linear_fc2.bias }
        + merger.linear_fc1.bias + merger.linear_fc2.bias   ==  110

Every other tensor (all vision weights, all text/mtp params) is present and
correct. A vision Linear running without its learned bias across all 27 encoder
blocks produces systematically distorted image features -> hallucinated vision,
while the text body stays at parity. That matches the observed symptom exactly.

Fix
---
Copy those 110 biases verbatim (BF16) from the BF16 source checkpoint into each
single-shard export. The export uses ONLY U8/BF16 safetensors dtypes with no
per-tensor metadata, so a full library round-trip is bit-lossless. We verify:
  * every PRE-EXISTING tensor is byte-identical before/after (sha256), and
  * every RESTORED bias byte-matches the source (sha256).
Only then do we swap the file in (rename-backup + atomic replace). Idempotent:
re-running finds 0 missing and skips.

Usage
-----
  python3 fix_vision_bias.py                 # all three R19j variants
  python3 fix_vision_bias.py --dry-run       # verify only, no swap
  python3 fix_vision_bias.py --targets DIR [DIR ...]
"""
import argparse
import gc
import hashlib
import json
import os
import struct
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Model area root: env-overridable (defaults = R19j box layout).
MODELS = os.environ.get("QUARK_MODELS_DIR", "/mnt/NVME2/AI/models")
SRC = os.path.join(MODELS, "huihui-ai/Huihui-Qwen3.8-27B-abliterated")

DEFAULT_TARGETS = [
    os.path.join(MODELS, "Huihui-Qwen3.8-27B-Quark-AWQ-MXFP4"),
    os.path.join(MODELS, "Huihui-Qwen3.8-27B-Quark-AWQ-MXFP4-MtpFp8"),
    os.path.join(MODELS, "Huihui-Qwen3.8-27B-Quark-AWQ-MXFP4-VisionQuant"),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def raw_bytes(t: torch.Tensor) -> bytes:
    """Canonical little-endian bytes of a tensor's storage (dtype-agnostic)."""
    t = t.detach().cpu().contiguous()
    return t.view(torch.uint8).numpy().tobytes()


def header_keys(path: str):
    """Fast key list + file metadata from a safetensors header (no tensor load)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.get("__metadata__")
    keys = [k for k in hdr if k != "__metadata__"]
    return keys, meta


def src_visual_biases():
    """All model.visual.* .bias keys in the BF16 source."""
    p = os.path.join(SRC, "model.safetensors")
    idx = os.path.join(SRC, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        files = {}
        out = []
        for k, v in wm.items():
            if k.startswith("model.visual") and k.endswith(".bias"):
                out.append(k)
        return out
    with safe_open(p, framework="pt") as sf:
        return [k for k in sf.keys()
                if k.startswith("model.visual") and k.endswith(".bias")]


def fix_target(target: str, dry_run: bool) -> bool:
    name = os.path.basename(target)
    orig = os.path.join(target, "model.safetensors")
    if not os.path.exists(orig):
        print(f"[{name}] SKIP: no model.safetensors")
        return True

    tgt_keys, _meta = header_keys(orig)
    tgt_set = set(tgt_keys)
    missing = [k for k in src_visual_biases() if k not in tgt_set]
    if not missing:
        print(f"[{name}] OK: all {len(src_visual_biases())} visual biases present "
              f"(nothing to do)")
        return True
    print(f"[{name}] restoring {len(missing)} missing visual biases "
          f"(of {len(tgt_keys)} existing tensors)")

    # --- load the 110 source biases (bf16) + their hashes ---
    src_p = os.path.join(SRC, "model.safetensors")
    src_idx = os.path.join(SRC, "model.safetensors.index.json")
    src_wm = None
    if os.path.exists(src_idx):
        src_wm = json.load(open(src_idx))["weight_map"]

    def src_tensor(key):
        if src_wm is None:
            with safe_open(src_p, framework="pt") as sf:
                return sf.get_tensor(key)
        with safe_open(os.path.join(SRC, src_wm[key]), framework="pt") as sf:
            return sf.get_tensor(key)

    biases = {}
    src_hash = {}
    for k in missing:
        t = src_tensor(k)
        assert t.dtype == torch.bfloat16, f"{k}: expected bf16, got {t.dtype}"
        biases[k] = t.contiguous()
        src_hash[k] = sha(raw_bytes(t))

    # --- load EVERY existing tensor, hash it (before) ---
    print(f"[{name}] loading {len(tgt_keys)} existing tensors ...")
    data = {}
    before = {}
    with safe_open(orig, framework="pt") as sf:
        for k in tgt_keys:
            t = sf.get_tensor(k)
            data[k] = t
            before[k] = sha(raw_bytes(t))
    print(f"[{name}] hashing done; merging biases -> writing temp file")

    data.update(biases)
    tmp = orig + ".new"
    save_file(data, tmp, metadata={"format": "pt"})
    del data
    gc.collect()

    # --- verify the written file ---
    vkeys, vmeta = header_keys(tmp)
    vset = set(vkeys)
    assert vmeta == {"format": "pt"}, f"file metadata changed: {vmeta}"
    assert len(vkeys) == len(tgt_keys) + len(missing), \
        f"key count {len(vkeys)} != {len(tgt_keys)}+{len(missing)}"
    bad_before = 0
    bad_new = 0
    with safe_open(tmp, framework="pt") as sf:
        for k in tgt_keys:                       # pre-existing must be unchanged
            if sha(raw_bytes(sf.get_tensor(k))) != before[k]:
                bad_before += 1
                print(f"[{name}]   MISMATCH pre-existing: {k}")
        for k in missing:                        # restored must match source
            if sha(raw_bytes(sf.get_tensor(k))) != src_hash[k]:
                bad_new += 1
                print(f"[{name}]   MISMATCH restored: {k}")
    if bad_before or bad_new:
        os.unlink(tmp)
        print(f"[{name}] VERIFY FAILED (pre-existing={bad_before}, "
              f"restored={bad_new}); left original untouched")
        return False
    print(f"[{name}] verified: {len(tgt_keys)} pre-existing byte-identical, "
          f"{len(missing)} biases match source")

    if dry_run:
        os.unlink(tmp)
        print(f"[{name}] DRY-RUN: would swap in place")
        return True

    bak = orig + ".bak_novisionbias"
    if not os.path.exists(bak):
        os.rename(orig, bak)
        print(f"[{name}] backup -> {os.path.basename(bak)}")
    os.replace(tmp, orig)
    print(f"[{name}] SWAPPED in place. Now {len(vkeys)} tensors.")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ok = True
    for t in args.targets:
        try:
            ok &= fix_target(t, args.dry_run)
        except Exception as e:
            ok = False
            print(f"[{os.path.basename(t)}] ERROR: {e!r}")
    print("\nRESULT:", "ALL OK" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()