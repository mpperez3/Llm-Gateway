#!/usr/bin/env bash
# Full deploy of the Qwen3.8-27B vLLM stack (with every speed patch) to a remote host.
#
#   LLM_SERVER_REMOTE_HOST=192.168.110.41 \
#   LLM_SERVER_REMOTE_USER=mpperez3 \
#   LLM_SERVER_REMOTE_PATH='$HOME/Heimdall-Gateway' \
#   MODELS_DIR=/scratch/tesla8/mpperez3/llm_models \
#   TP=2 \
#   bash tools/deploy_vllm_stack.sh
#
# It rsyncs this checkout (the code, not the models), then on the remote host:
#   1. uv sync
#   2. install the gateway (user mode, vLLM backend) if `llm-server` is missing
#   3. apply the bundled vLLM patches (tools/apply_vllm_patch.sh)
#   4. download the W4A16 model and the DFlash2 drafter
#   5. register the model and write the tuned catalog overrides (DFlash2 + lookup)
#   6. build the MTP draft vocab (kept for the MTP fallback)
#   7. render config, restart services and verify the API
set -euo pipefail

REMOTE_HOST="${LLM_SERVER_REMOTE_HOST:?Set LLM_SERVER_REMOTE_HOST}"
REMOTE_USER="${LLM_SERVER_REMOTE_USER:-${USER}}"
REMOTE_PATH="${LLM_SERVER_REMOTE_PATH:?Set LLM_SERVER_REMOTE_PATH (code dir on the remote)}"
MODELS_DIR="${MODELS_DIR:-/scratch/tesla8/mpperez3/llm_models}"
TP="${TP:-2}"
MODEL_ID='Qwen3.8-27B-AutoRound-INT4-[vLLM]'
HF_REPO='dbirks/Qwen3.8-27B-W4A16-AutoRound'
DRAFTER='syvai/Qwen3.8-27B-DFlash2-W4A16'
MODEL_DIR="$MODELS_DIR/dbirks/Qwen3.8-27B-W4A16-AutoRound"
DRAFT_DIR="$MODELS_DIR/syvai/Qwen3.8-27B-DFlash2-W4A16"

echo "== 1/2 sync code -> $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -avz \
  --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
  --exclude '*.log' --exclude 'models' \
  ./ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

echo "== 2/2 bootstrap on the remote"
ssh "$REMOTE_USER@$REMOTE_HOST" bash -s <<REMOTE
set -euo pipefail
cd "$REMOTE_PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
uv sync

if ! command -v llm-server >/dev/null 2>&1; then
  uv run llm-server install --mode user --backend vllm-beta \
    --models-dir "$MODELS_DIR" --public-host 0.0.0.0 --public-port 11436 \
    --no-api-auth --no-api-https --idle-ttl 300
fi

# vLLM entrypoint + PATH for the llama-swap service (the installer does not
# always create it), and the inline-system chat template for prefix caching.
install -m 0755 llamacpp_stack/bundle/vllm/vllm-server "$HOME/.local/bin/vllm-server"
mkdir -p "$HOME/.config/systemd/user/llm-server-manager.service.d" \
         "$HOME/.config/systemd/user/llm-server-router.service.d"
printf '[Service]\nEnvironment=VLLM_SERVER_BIN=%s/.local/bin/vllm-server\nEnvironment=PATH=%s/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n' \
  "$HOME" "$HOME" | tee \
  "$HOME/.config/systemd/user/llm-server-manager.service.d/vllm-server-bin.conf" \
  "$HOME/.config/systemd/user/llm-server-router.service.d/vllm-server-bin.conf" >/dev/null
systemctl --user daemon-reload
systemctl --user restart llm-server-manager
cp llamacpp_stack/bundle/vllm/qwen38_chat_template_inline_system.jinja \
   "$MODEL_DIR/chat_template_inline_system.jinja"

bash tools/apply_vllm_patch.sh

uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", local_dir="$MODEL_DIR")
snapshot_download("$DRAFTER", local_dir="$DRAFT_DIR",
                  allow_patterns=["*.json", "*.safetensors", "README.md"])
print("models ready")
PY

uv run llm-server add "$HF_REPO" --model-id "$MODEL_ID" --ctx-size 262144 || true

uv run python - <<'PY'
import json, os
p = os.path.expanduser("~/.local/state/llm-server/catalog.json")
d = json.load(open(p))
models = d.get("models") if isinstance(d, dict) else d
for m in models or []:
    if m.get("model_id") != "$MODEL_ID":
        continue
    m["backend"] = "vllm"
    m["local_path"] = "$MODEL_DIR"
    m["ctx_size"] = 262144
    m["load_capabilities"] = ["text-to-text", "image-text-to-text"]
    m["server_overrides"] = {"replicas": 0, "vllm": {
        "max_model_len": 262144,
        "tensor_parallel_size": $TP,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 1024,
        "kv_cache_dtype": "fp8",
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": True,
        "max_cudagraph_capture_size": 16,
        "trust_remote_code": True,
        "speculative_config": {
            "method": "dflash",
            "model": "$DRAFT_DIR",
            "num_speculative_tokens": 7,
            "draft_sample_method": "probabilistic",
        },
        "reasoning_parser": "qwen3",
        "enable_auto_tool_choice": True,
        "tool_call_parser": "qwen3_xml",
        "limit_mm_per_prompt": {"image": 2, "video": 0},
        "mamba_cache_mode": "align",
        "enable_per_request_metrics": True,
        "enable_force_include_usage": True,
        "chat_template": "$MODEL_DIR/chat_template_inline_system.jinja",
    }}
tmp = p + ".tmp"
json.dump(d, open(tmp, "w"), indent=1, ensure_ascii=False)
os.replace(tmp, p)
print("catalog overrides written")
PY

uv run llm-server vllm-draft-vocab --model "$MODEL_ID" || true
uv run llm-server update
systemctl --user restart llm-server-manager llm-server-router
sleep 6
curl -fsS http://127.0.0.1:11436/health >/dev/null && echo "swap: OK"
curl -fsS http://127.0.0.1:11435/v1/models >/dev/null && echo "gateway: OK"
REMOTE

echo "== done"
