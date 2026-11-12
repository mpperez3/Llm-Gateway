#!/usr/bin/env python3
"""Banco de pruebas con prompts reales del proyecto.

Dos cargas:
  fixture  petición real de 481 items capturada en
           tests/fixtures/gateway_raw_examples/extra_long_qwen_481_raw.json
           (~165k tokens de contexto)
  recipe   los 8 prompts de bench/prompts_real.jsonl de la receta
           syv-ai/qwen38-27b-rtx3090 (C1 corto, 1024 tokens de salida)

Mide por el camino del cliente (gateway/guard), con thinking activado por defecto
y >=1024 tokens de salida, que es donde la aceptación de spec decode es realista.

Ejemplos:
  python tools/bench_real_prompts.py --workload recipe --repeats 2
  python tools/bench_real_prompts.py --workload fixture --repeats 2
  python tools/bench_real_prompts.py --workload both --base-url http://127.0.0.1:11436
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "gateway_raw_examples" / "extra_long_qwen_481_raw.json"
DEFAULT_RECIPE_PROMPTS = Path("/tmp/prompts_real.jsonl")
RECIPE_URL = "https://raw.githubusercontent.com/xvus/syv-ai-qwen38-27b-rtx3090/main/bench/prompts_real.jsonl"
FIXTURE_CHAR_CAP = 600_000


def fixture_prompt(cap: int = FIXTURE_CHAR_CAP) -> str:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chunks: list[str] = []
    total = 0
    for item in payload.get("input") or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text") or part.get("content") or "") if isinstance(part, dict) else str(part)
                for part in content
            )
        text = str(content or "")
        if not text.strip():
            continue
        chunks.append(f"\n\n[{item.get('role') or 'user'}]\n{text}")
        total += len(text)
        if total >= cap:
            break
    prompt = "".join(chunks)[:cap]
    return prompt + "\n\nResume que hace este proyecto y donde estan los riesgos principales."


def recipe_prompts(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(
            f"faltan los prompts de la receta en {path}\n"
            f"descargalos con: curl -s {RECIPE_URL} -o {path}"
        )
    return [json.loads(line)["prompt"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def call(url: str, model: str, prompt: str, max_tokens: int, temperature: float, thinking: bool) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if thinking:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read()[:200].decode(errors='replace')}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    timings = payload.get("timings") or {}
    usage = payload.get("usage") or {}
    return {
        "wall_s": round(time.time() - started, 1),
        "prompt_n": int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0),
        "predicted_n": int(timings.get("predicted_n") or usage.get("completion_tokens") or 0),
        "prompt_tps": round(float(timings.get("prompt_per_second") or 0), 1),
        "decode_tps": round(float(timings.get("predicted_per_second") or 0), 2),
        "finish": (payload.get("choices") or [{}])[0].get("finish_reason"),
    }


def summarize(label: str, runs: list[dict], max_tokens: int) -> dict:
    valid = [r for r in runs if "error" not in r and r["predicted_n"] >= max_tokens * 0.9]
    return {
        "workload": label,
        "runs": len(runs),
        "valid": len(valid),
        "median_decode_tps": round(statistics.median(r["decode_tps"] for r in valid), 2) if valid else None,
        "mean_decode_tps": round(statistics.mean(r["decode_tps"] for r in valid), 2) if valid else None,
        "prompt_tokens": sorted({r["prompt_n"] for r in valid}),
        "details": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", choices=["fixture", "recipe", "both"], default="both")
    parser.add_argument("--base-url", default="http://127.0.0.1:11436")
    parser.add_argument("--model", default="Qwen3.8-27B-AutoRound-INT4-[vLLM]")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-thinking", action="store_true", help="solo para preguntas sobre thinking off")
    parser.add_argument("--recipe-prompts", default=str(DEFAULT_RECIPE_PROMPTS))
    args = parser.parse_args()

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    thinking = not args.no_thinking
    report: list[dict] = []

    if args.workload in ("fixture", "both"):
        prompt = fixture_prompt()
        print(f"[fixture] {len(prompt)} chars, {args.repeats} corrida(s), thinking={thinking}", file=sys.stderr)
        runs = [call(url, args.model, prompt, args.max_tokens, args.temperature, thinking) for _ in range(args.repeats)]
        report.append(summarize("fixture", runs, args.max_tokens))

    if args.workload in ("recipe", "both"):
        prompts = recipe_prompts(Path(args.recipe_prompts))
        runs = []
        for i, prompt in enumerate(prompts):
            print(f"[recipe] prompt {i + 1}/{len(prompts)}", file=sys.stderr)
            runs.append(call(url, args.model, prompt, args.max_tokens, args.temperature, thinking))
        report.append(summarize("recipe", runs, args.max_tokens))

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
