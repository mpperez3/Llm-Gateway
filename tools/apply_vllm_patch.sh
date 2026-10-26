#!/usr/bin/env bash
# Apply (or revert) the bundled vLLM patches for Qwen3.5 on 24 GB cards.
#
#   tools/apply_vllm_patch.sh            # apply every patch not applied yet
#   tools/apply_vllm_patch.sh --revert   # revert every applied patch
#
# vLLM is a wheel inside the gateway venv, so this edits site-packages in place
# with timestamped .bak-* copies. Re-run after any vLLM reinstall/upgrade.
set -euo pipefail

VENV="${LLM_SERVER_VENV:-$HOME/.local/opt/llm-server/venv}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$ROOT/llamacpp_stack/bundle/patches-vllm"
MODE="${1:-apply}"

[ -d "$PATCH_DIR" ] || { echo "no patch dir: $PATCH_DIR" >&2; exit 1; }
SITE_PACKAGES="$("$VENV/bin/python" -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
cd "$SITE_PACKAGES"

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
[ ${#patches[@]} -gt 0 ] || { echo "no patches found"; exit 0; }

rc=0
for p in "${patches[@]}"; do
    name="$(basename "$p")"
    if [[ "$MODE" == "--revert" ]]; then
        if patch -R -p1 --dry-run --silent < "$p" >/dev/null 2>&1; then
            patch -R -p1 < "$p"
            echo "reverted: $name"
        else
            echo "not applied: $name"
        fi
        continue
    fi
    if patch -p1 --dry-run --silent < "$p" >/dev/null 2>&1; then
        grep -oE '^\+\+\+ b/[^[:space:]]+' "$p" | sed 's|^+++ b/||' | while read -r f; do
            [ -f "$f" ] && cp -a "$f" "$f.bak-$(date +%Y%m%d-%H%M%S)"
        done
        patch -p1 < "$p"
        echo "applied: $name"
    elif patch -R -p1 --dry-run --silent < "$p" >/dev/null 2>&1; then
        echo "already applied: $name"
    else
        echo "FAILED (neither applies nor reverts cleanly): $name" >&2
        rc=1
    fi
done

for f in vllm/model_executor/models/qwen3_5.py vllm/model_executor/models/qwen3_5_mtp.py; do
    [ -f "$f" ] && "$VENV/bin/python" -m py_compile "$f"
done
echo "compile OK (restart the vLLM engine to take effect)"
exit $rc
