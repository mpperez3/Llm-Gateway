#!/usr/bin/env python3
"""One-shot benchmark client for any OpenAI-compatible llama.cpp/EXL3 server.

Sends a single cold prompt (unique nonce, so the prefix cache cannot answer it)
and records the **server-side** timings: `prompt_n`, `prompt_per_second`,
`predicted_n`, `predicted_per_second`, plus draft counters when MTP is active.

Why server-side: `completion_tokens / wall-clock` includes template, proxy and
HTTP and reads 15-20% lower than pure decode time.

See `docs/PERF-BASELINE-4090-QWEN38-27B.md` for the measurement protocol
(warm-up, fixed `--seed` when comparing draft settings, 2+ reps).

Usage:

    python tools/bench_chat_timings.py --port 18080 --prompt-tokens 44000 --gen 192
    python tools/bench_chat_timings.py --port 8080 --prompt-tokens 89000 --gen 384 \\
        --seed 424242 --out /tmp/bench.jsonl

Appends one JSON line per call to `--out` and prints the same record.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import time
import urllib.request

WORDS = (
    "model inference kernel bandwidth memory scheduler latency throughput quantized "
    "attention tensor shard batch pipeline cache vector matrix compute arc gateway "
    "replica context prompt token stream decode prefill driver adapter"
).split()


def build_prompt(target_tokens: int, seed: int, task: str = "summarize") -> str:
    """Deterministic pseudo-prose of roughly `target_tokens` tokens.

    The first line carries a random nonce so two calls never share a prefix;
    without it the server's prefix/KV cache turns prefill into a cache hit and
    the measurement is meaningless.

    `task="quote"` appends an instruction to reproduce a passage verbatim: that
    is the regime where lookup/n-gram speculative decoding fires, so it is the
    way to measure `--spec-type ...ngram-simple/copyspec`.
    """
    rnd = random.Random(seed)
    nonce = "".join(rnd.choice(string.ascii_letters + string.digits) for _ in range(24))
    head = f"Session marker {nonce}. Read the following engineering log and then answer.\n\n"
    target_chars = int(target_tokens * 3.6)  # ~3.6 chars/token for English prose
    body: list[str] = []
    size = 0
    i = 0
    while size < target_chars:
        i += 1
        line = (
            f"Entry {i:06d}: "
            + " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(12, 22)))
            + ".\n"
        )
        body.append(line)
        size += len(line)
    if task == "quote":
        text = "".join(body)
        passage = text[-max(4000, len(text) // 2):]
        tail = (
            "\nPassage to reproduce verbatim:\n<<<\n"
            + passage
            + "\n>>>\nOutput the passage again, character for character, with no "
            "commentary, no markdown and no preamble.\nAnswer:"
        )
    else:
        tail = "\nQuestion: summarize the main theme of the log above in one sentence.\nAnswer:"
    return head + "".join(body) + tail


def post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="bench", help="value for the request's model field")
    ap.add_argument("--prompt-tokens", type=int, required=True, help="approximate size")
    ap.add_argument("--gen", type=int, default=192, help="max_tokens to generate")
    ap.add_argument("--label", default="run")
    ap.add_argument("--task", default="summarize", choices=["summarize", "quote"],
                    help="quote: reproduce a passage verbatim (exercise ngram/copyspec drafting)")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix it to compare settings: same seed -> same output -> same draft acceptance")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", default="/tmp/llmperf/results/bench.jsonl")
    a = ap.parse_args()

    seed = a.seed if a.seed is not None else int(time.time() * 1000) % 10**9
    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": build_prompt(a.prompt_tokens, seed, a.task)}],
        "max_tokens": a.gen,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False,
    }
    t0 = time.time()
    try:
        body = post(f"http://{a.host}:{a.port}/v1/chat/completions", payload, a.timeout)
        err = None
    except Exception as exc:  # noqa: BLE001 - a failed run is data, not a crash
        body, err = None, f"{type(exc).__name__}: {exc}"

    rec = {
        "label": a.label,
        "task": a.task,
        "target_prompt_tokens": a.prompt_tokens,
        "gen": a.gen,
        "seed": seed,
        "wall_s": round(time.time() - t0, 3),
        "error": err,
    }
    if body:
        tim = body.get("timings") or {}
        usage = body.get("usage") or {}
        rec.update({
            "prompt_n": tim.get("prompt_n", usage.get("prompt_tokens")),
            "prompt_ms": tim.get("prompt_ms"),
            "prompt_tps": tim.get("prompt_per_second"),
            "predicted_n": tim.get("predicted_n", usage.get("completion_tokens")),
            "predicted_ms": tim.get("predicted_ms"),
            "decode_tps": tim.get("predicted_per_second"),
            "cache_n": tim.get("cache_n"),
            "draft_n": tim.get("draft_n"),
            "draft_acc": tim.get("draft_n_accepted"),
        })

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return 0 if not rec["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
