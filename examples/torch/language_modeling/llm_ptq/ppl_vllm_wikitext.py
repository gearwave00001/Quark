#!/usr/bin/env python3
"""ppl_vllm_wikitext.py — wikitext-2-raw-v1 PPL against a DEPLOYED vLLM instance.

Replicates quark's ppl_eval metric EXACTLY (quark/contrib/llm_eval/evaluation.py)
so the number is directly comparable to the <=6.51 gate:

  - data:    load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test"),
             text = "\\n\\n".join(all lines), tokenized ONCE with the model's tokenizer
  - windows: NON-OVERLAPPING 2048-token chunks (trailing <2048 tokens discarded)
  - scoring: causal NLL at every position inside each window (position 0 unscored)
  - ppl:     exp( sum(all position NLLs) / (nsamples * 2047) )
             (quark multiplies the per-window MEAN loss by 2048 then divides the
              sum by nsamples*2048 — the factors cancel, leaving the plain
              per-position mean NLL)

Mechanism: vLLM's OpenAI-compatible /v1/completions endpoint with
prompt_logprobs=0, which returns the logprob of the ACTUAL token at every prompt
position. Needs only `requests`, `datasets`, `transformers` (all present in any
vLLM image). Run it inside the vLLM container (or anywhere with those packages +
network access to the served endpoint).

Usage:
  BASE_URL=http://localhost:8000 \\
  python3 ppl_vllm_wikitext.py --model_dir /path/to/exported/model [--model NAME]

Notes:
  - Pass raw token-id lists as prompts: no chat template is applied, matching
    quark which feeds raw input_ids to the model.
  - Text PPL is vision-tower-independent, so either model variant (bf16-vision
    parity or vision-quantized) yields the same number.
"""
import argparse
import math
import os
import time

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

SEQLEN = 2048


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_dir", required=True, help="exported model dir (used for the tokenizer)")
    ap.add_argument("--base_url", default=os.environ.get("BASE_URL", "http://localhost:8000"))
    ap.add_argument("--model", default=os.environ.get("MODEL", ""), help="served model name/path (default: model_dir)")
    ap.add_argument("--batch", type=int, default=16, help="windows per API call")
    ap.add_argument("--timeout", type=int, default=3600, help="per-request timeout (s)")
    ap.add_argument("--sleep", type=float, default=3.0, help="pause (s) between batches to avoid overloading the server")
    ap.add_argument("--retries", type=int, default=4, help="attempts per batch (exponential backoff on 5xx/timeout)")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    model_name = args.model or args.model_dir

    # ---- identical data prep to quark evaluation.py (load + tokenize once) ----
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(testdata["text"])
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    ids = tok(text, return_tensors="pt").input_ids.flatten().tolist()
    nsamples = len(ids) // SEQLEN
    chunks = [ids[i * SEQLEN : (i + 1) * SEQLEN] for i in range(nsamples)]
    print(
        "[PPL-VLLM] %d tokens -> %d x %d windows; base=%s model=%s"
        % (len(ids), nsamples, SEQLEN, base, model_name),
        flush=True,
    )

    total_nll = 0.0
    scored = 0
    t0 = time.time()
    for b in range(0, nsamples, args.batch):
        grp = chunks[b : b + args.batch]
        payload = {
            "model": model_name,
            "prompt": grp,  # list of int-id lists: raw tokens, no template
            "prompt_logprobs": 0,  # logprob of the ACTUAL token at each position
            "max_tokens": 1,  # generated token ignored
            "temperature": 0,
        }
        data = None
        for attempt in range(1, args.retries + 1):
            try:
                r = requests.post(base + "/v1/completions", timeout=args.timeout, json=payload)
                r.raise_for_status()
                data = r.json()
                break
            except requests.RequestException as exc:
                if attempt == args.retries:
                    raise
                wait = min(60.0, 5.0 * (2 ** (attempt - 1)))
                print(
                    "[PPL-VLLM] windows %d-%d attempt %d/%d failed (%s); retrying in %.0fs"
                    % (b // args.batch, (b + len(grp) - 1) // args.batch, attempt, args.retries, exc, wait),
                    flush=True,
                )
                time.sleep(wait)
        for ch in data["choices"]:
            plps = ch.get("prompt_logprobs") or []
            for pos, entry in enumerate(plps):
                if pos == 0 or not entry:
                    continue  # position 0 has no left context
                # prompt_logprobs=0 -> exactly one entry: the actual token
                lp = next(iter(entry.values()))["logprob"]
                total_nll += -lp
                scored += 1
        done = min(b + args.batch, nsamples)
        print("[PPL-VLLM] %d/%d windows (%.0fs)" % (done, nsamples, time.time() - t0), flush=True)
        if done < nsamples:
            time.sleep(args.sleep)

    expected = nsamples * (SEQLEN - 1)
    if scored != expected:
        print("[PPL-VLLM] WARNING: scored %d positions, expected %d" % (scored, expected), flush=True)
    ppl = math.exp(total_nll / scored)
    # Same line format as quark's eval_model so downstream checks pick it up identically.
    print("[INFO] Perplexity: %s" % ppl)
    print("[PPL-VLLM] positions=%d elapsed=%.0fs" % (scored, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()