#!/usr/bin/env python3
"""Enable vLLM's native thinking budget with a forced reasoning close.

vLLM 0.32 already implements the budget (`thinking_token_budget` per request,
`ReasoningConfig` on the server). What it does not do by default is *close*
the thinking block and push the model to answer: that is
`reasoning_end_str`, a server-side string that vLLM tokenizes and forces when
the budget is exhausted.

The gateway sends `thinking_token_budget` when the family/model policy is
`reasoning_budget: half_context` (see `resolve_request_reasoning_budget`), so
the only per-model change left is this server flag. It is stored in the
catalog under `server_overrides.vllm.reasoning_config`, which survives
`llm-server update` re-renders.

Idempotent: running it twice leaves the catalog unchanged.

Examples
--------
    # show vLLM models and whether they already carry the forced close
    python tools/patch_vllm_reasoning.py --list

    # dry run (default): print the JSON that would be written
    python tools/patch_vllm_reasoning.py

    # write the catalog and re-render + restart the services
    python tools/patch_vllm_reasoning.py --apply

    # system installation, single model, keep the services running
    python tools/patch_vllm_reasoning.py --system --model 'Qwen3.8-27B-AutoRound-INT4-[vLLM]' \
        --apply --no-restart
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MESSAGE = "Okay, I have thought enough. I will now provide the final answer."
DEFAULT_END_MARKER = "</think>"
USER_CATALOG = Path.home() / ".local/state/llm-server/catalog.json"
SYSTEM_CATALOG = Path("/var/lib/llm-server/catalog.json")


def default_catalog(system: bool) -> Path:
    return SYSTEM_CATALOG if system else USER_CATALOG


def load_catalog(path: Path) -> tuple[dict, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, ("models" if isinstance(data, dict) and "models" in data else "list")


def iter_models(data: dict, shape: str) -> list[dict]:
    models = data["models"] if shape == "models" else data
    return models if isinstance(models, list) else []


def model_backend(model: dict) -> str:
    overrides = model.get("server_overrides") or {}
    if isinstance(overrides, dict) and str(overrides.get("engine") or "").strip():
        return str(overrides.get("engine"))
    return str(model.get("backend") or "llama.cpp")


def build_end_str(message: str, end_marker: str) -> str:
    return f"\n\n{message}\n{end_marker}\n\n"


def patch_model(model: dict, end_str: str) -> bool:
    overrides = model.setdefault("server_overrides", {})
    vllm = overrides.setdefault("vllm", {})
    config = vllm.setdefault("reasoning_config", {})
    if config.get("reasoning_end_str") == end_str:
        return False
    config["reasoning_end_str"] = end_str
    return True


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", type=Path, default=None, help="catalog.json path (default: mode-based)")
    parser.add_argument("--system", action="store_true", help="use the system-mode catalog and services")
    parser.add_argument("--model", action="append", default=[], metavar="MODEL_ID",
                        help="model id to patch (repeatable, default: every vLLM model)")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="transition phrase forced before the end marker")
    parser.add_argument("--end-marker", default=DEFAULT_END_MARKER,
                        help="reasoning end marker of the family (default: </think>)")
    parser.add_argument("--list", action="store_true", help="list vLLM models and their current state, then exit")
    parser.add_argument("--apply", action="store_true", help="write the catalog, run `llm-server update` and restart")
    parser.add_argument("--no-restart", action="store_true", help="with --apply: do not restart the services")
    parser.add_argument("--dry-run", action="store_true", help="force dry-run even with --apply")
    args = parser.parse_args()

    catalog = args.catalog or default_catalog(args.system)
    if not catalog.exists():
        print(f"catalog not found: {catalog}", file=sys.stderr)
        return 2

    data, shape = load_catalog(catalog)
    models = iter_models(data, shape)
    vllm_models = [m for m in models if model_backend(m).lower() == "vllm"]
    end_str = build_end_str(args.message, args.end_marker)

    if args.list or not args.apply or args.dry_run:
        print(f"catalog: {catalog}")
        print(f"reasoning_end_str: {end_str!r}")
        for model in vllm_models:
            current = (((model.get("server_overrides") or {}).get("vllm") or {}).get("reasoning_config") or {})
            state = "patched" if current.get("reasoning_end_str") == end_str else (
                "different" if current.get("reasoning_end_str") else "missing")
            print(f"  [{state:9s}] {model.get('model_id')}")
        if args.list:
            return 0
        if not args.apply or args.dry_run:
            print("\ndry-run: nothing written (pass --apply to write)")
            return 0

    wanted = set(args.model)
    targets = [m for m in vllm_models if not wanted or m.get("model_id") in wanted]
    if wanted:
        missing = wanted - {m.get("model_id") for m in targets}
        for name in sorted(missing):
            print(f"model not found or not vllm: {name}", file=sys.stderr)
        if missing:
            return 2
    if not targets:
        print("no vLLM models selected", file=sys.stderr)
        return 2

    changed = [m.get("model_id") for m in targets if patch_model(m, end_str)]
    if not changed:
        print("catalog already patched; nothing to do")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = catalog.with_name(f"{catalog.name}.bak-reasoning-{stamp}")
    shutil.copy2(catalog, backup)
    catalog.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"patched: {', '.join(changed)}")
    print(f"backup : {backup}")

    if run(["llm-server", "update"]) != 0:
        print("`llm-server update` failed; the catalog is patched, fix the error and re-run update", file=sys.stderr)
        return 1
    if not args.no_restart:
        systemctl = ["systemctl"]
        if not args.system:
            systemctl.append("--user")
        run([*systemctl, "restart", "llm-server-manager", "llm-server-router"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
