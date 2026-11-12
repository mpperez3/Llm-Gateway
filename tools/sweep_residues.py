#!/usr/bin/env python3
"""Barrido de residuos mod 128 (protocolo de la receta) contra el gateway.

Cada prompt se envía DOS veces: la segunda es un self-hit de prefix cache, que es
la condición que necesita el bug. Se juzga por cobertura verbatim del documento,
nunca por la firma del fallo.

Uso: python3 tools/sweep_residues.py <label> [pads] [start]
"""
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:11436"
MODEL = "Qwen3.8-27B-AutoRound-INT4-[vLLM]"
DOC = open("/tmp/fixture_ctx.txt", encoding="utf-8").read()[:30000]  # corpus de la fixture real
WINDOW = 40


def coverage(answer: str, doc: str) -> float:
    if len(answer) < WINDOW:
        return 0.0
    windows = [answer[i:i + WINDOW] for i in range(0, len(answer) - WINDOW, WINDOW)]
    hits = sum(1 for w in windows if w in doc)
    return hits / len(windows) if windows else 0.0


def once(content: str) -> tuple[int, str]:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = json.loads(response.read())
    timings = payload.get("timings") or {}
    usage = payload.get("usage") or {}
    prompt_n = int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0)
    answer = (payload["choices"][0]["message"].get("content") or "")
    return prompt_n, answer


def main() -> None:
    label = sys.argv[1]
    pads = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rows = []
    for pad in range(start, start + pads):
        content = ("Dokument:\n\n" + DOC +
                   "\n\nGengiv ordret de første 60 linjer af dokumentet. "
                   "Ingen kommentarer, kun teksten." + " x" * pad)
        _, _ = once(content)
        prompt_n, answer = once(content)
        cov = coverage(answer, DOC)
        rows.append({"pad": pad, "residue": prompt_n % 128, "prompt_n": prompt_n,
                     "chars": len(answer), "coverage": round(cov, 3),
                     "preview": answer[:60]})
        flag = "BROKEN" if (len(answer) < 40 or cov < 0.5) else "ok"
        print(f"[{label}] pad={pad:3d} residue={prompt_n % 128:3d} len={prompt_n:6d} "
              f"chars={len(answer):4d} cov={cov:.2f} {flag}", flush=True)
        if flag == "BROKEN":
            print(f"    preview={answer[:120]!r}", flush=True)
    broken = [r for r in rows if r["chars"] < 40 or r["coverage"] < 0.5]
    print(json.dumps({
        "label": label,
        "pads": pads,
        "broken": len(broken),
        "median_coverage": round(statistics.median(r["coverage"] for r in rows), 3),
        "broken_residues": [r["residue"] for r in broken],
        "broken_detail": broken[:10],
    }, indent=1))


if __name__ == "__main__":
    main()
