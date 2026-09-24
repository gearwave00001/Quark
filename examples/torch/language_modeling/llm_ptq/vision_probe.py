#!/usr/bin/env python3
"""vision_probe.py — end-to-end vision check against a DEPLOYED vLLM instance.

Posts an image + a captioning prompt to /v1/chat/completions (OpenAI-compatible
multimodal) and prints the model's description. Used to confirm the R19j MXFP4
exports' vision tower works after restoring the 110 dropped model.visual.* biases.

Self-contained: if --image is not given it synthesizes a distinctive test image
with PIL (colored geometric shapes + large digits) so a working ViT must name the
actual colors/shapes/numbers, while a broken one hallucinates.

Usage:
  BASE_URL=http://localhost:5678 python3 vision_probe.py
  BASE_URL=http://localhost:5678 python3 vision_probe.py --image /path/to/photo.jpg
  python3 vision_probe.py --prompt "What single object is in this image?"
"""
import argparse
import base64
import io
import os
import sys

import requests


def gen_test_image(path: str) -> None:
    """Draw an unambiguous composite: red circle, blue square, green triangle,
    plus large digits '7 3'. A healthy vision encoder reports these; a broken
    one (dropped biases) garbles them."""
    from PIL import Image, ImageDraw
    W = H = 512
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([60, 60, 220, 220], fill="red")                 # red circle TL
    d.rectangle([300, 80, 440, 220], fill="blue")             # blue square TR
    d.polygon([(90, 460), (230, 460), (160, 320)], fill="green")  # green tri BL
    d.text((320, 360), "73", fill="black")                     # digits BR
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_url", default=os.environ.get("BASE_URL", "http://localhost:5678"))
    ap.add_argument("--image", default="", help="path to an image; if empty, synthesize one")
    ap.add_argument("--prompt", default="Describe exactly what you see in this image, "
                    "including any shapes, colors, and numbers.")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    # discover the served model id rather than hardcoding it
    r = requests.get(base + "/v1/models", timeout=30)
    r.raise_for_status()
    models = [m["id"] for m in r.json().get("data", [])]
    if not models:
        print("[vision-probe] no models served at %s/v1/models" % base); sys.exit(2)
    model_id = models[0]
    print("[vision-probe] base=%s served_model=%s" % (base, model_id))

    img_path = args.image
    if not img_path:
        img_path = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "vision_probe_test.png")
        gen_test_image(img_path)
        print("[vision-probe] synthesized test image -> %s" % img_path)
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(img_path)[1].lstrip(".").lower() or "png"
    data_url = "data:image/%s;base64,%s" % (ext, b64)

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": args.prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    print("[vision-probe] POST /v1/chat/completions ...")
    r = requests.post(base + "/v1/completions".replace("/completions", "/chat/completions"),
                      timeout=args.timeout, json=payload)
    r.raise_for_status()
    j = r.json()
    text = j["choices"][0]["message"]["content"]
    print("\n==================== MODEL RESPONSE ====================")
    print(text)
    print("=========================================================")


if __name__ == "__main__":
    main()