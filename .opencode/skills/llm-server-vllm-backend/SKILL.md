---
name: llm-server-vllm-backend
description: "Operación del backend vLLM (beta) en LLM Server: parches, reasoning parity, draft vocab, métricas de speculative y naming \"[vLLM]\". Úsala SIEMPRE que se añada o depure un modelo vLLM, cuando falten `timings`/`Drafted`, cuando una respuesta se corte al primer token, cuando haya que reaplicar parches tras un upgrade, o cuando se compare vLLM contra buun/llama.cpp. No la uses para modelos GGUF ni EXL3."
---

# Backend vLLM

vLLM no es "otro llama.cpp": tiene su propio modelo de parches, sus propias métricas y su
propio formato de stream. Aplicar aquí recetas de llama.cpp es el error más repetido.

## 1. Estado y verificación rápida

```bash
uv run llm-server config-keys --format json | jq '.vllm'
llm-server ps | grep -i vllm
curl -s http://127.0.0.1:11435/v1/models | jq '.data[] | select(.id|test("\\[vLLM\\]")) | {id, context_length}'
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

Todo modelo vLLM debe aparecer en la API con el sufijo **`[vLLM]`** en su id.
Si no aparece: no está registrado, o el backend no arrancó (`500 failed to start command`).

## 2. Flags: qué hace y qué ignora

| Flag | vLLM |
|---|---|
| `--ctx-size N` | → `--max-model-len N` |
| `--dtype` | precisión de inferencia |
| `--n-gpu-layers` | **ignorado** (usa todas las capas) |
| `--fit`, `--fitc`, `--fitt`, `--mirostat` | **ignorados** |
| `batch-size` / `ubatch-size` | **no existen**; usar `max_num_seqs` |
| `keep`, `cache-ram`, `ctx_checkpoints` | **no aplican** (son de llama.cpp/EXL3) |
| `compilation_config` | sí; PIECEWISE se midió **más lento** a contexto largo |

Knobs útiles ya disponibles en catálogo: `max_num_seqs` (C8 agregado ≫ C1),
`draft_sample_method: probabilistic` (importa con `temperature > 0`),
`mamba_cache_mode: align` (híbridos + prefix caching),
`enable_per_request_metrics` + `enable_force_include_usage`.

## 3. Parches: el inventario es el contrato

Inventario canónico: **`llamacpp_stack/bundle/patches-vllm/README.md`**.
Aplicados sobre vLLM 0.32: `qwen35-embed-quant`, `qwen35-mtp-share-heads`,
`response-timings`, `slot-print-timing`, `qwen35-mtp-draft-vocab`.

```bash
bash tools/apply_vllm_patch.sh          # reaplicar tras CADA upgrade de vLLM
uv run llm-server vllm-draft-vocab --model '<MODEL_ID>'   # vocab truncado (~40 960 ids)
```

- **Tras cada upgrade de vLLM hay que reaplicar los parches.** Un upgrade silencioso los
  pierde y con ellos `timings`, `slot print_timing` y el draft vocab.
- El vocab se genera por modelo (`tools/build_draft_vocab.py <model_dir> --ids
  tools/draft_vocab_ids.json`) y también se autogenera al añadir un modelo vLLM con pesos
  `mtp.*` (`LLM_SERVER_DRAFT_VOCAB=0` lo desactiva).
- Pendientes (con fuente en `docs/VLLM-BETA.md`): `spec-decode-attn`, sampler fast-softmax,
  lm_head/MTP int4 GPTQ, W4A8 Marlin, int4 KV per-token-head, DFlash2 (no portado a 0.32).
  **No los presentes como disponibles**: están pendientes.

## 4. Reasoning parity

- El campo por request es `thinking_token_budget` (no `thinking_budget_tokens`).
- El cierre forzado va en el **override del modelo**, no en un default global:
  `server_overrides.vllm.reasoning_config.reasoning_end_str` (el marcador es de familia).
- El gateway renombra `delta.reasoning` → `reasoning_content` y **separa** el frame que
  mezcla `content` + `reasoning`. Si ves respuestas cortadas al primer token, revisa esa
  separación antes que nada.
- `reasoning_effort` es del cliente: no lo sobrescribas.

Reaplicar con el helper idempotente (dry-run por defecto):

```bash
python tools/patch_vllm_reasoning.py --list
python tools/patch_vllm_reasoning.py --model '<MODEL_ID>' --apply
```

## 5. Métricas

| Dato | vLLM |
|---|---|
| `usage.prompt_tokens` / `completion_tokens` | sí, nativo |
| `timings` (prefill/decode) | solo con el parche `response-timings` |
| `slot print_timing` en journal | **nunca** |
| `draft_n` / `Drafted` | solo con speculative **y** `per_request_spec_decode_metrics` (`detailed`/`summary`) |

El gateway comprueba con `vllm-server --help | grep per-request-spec-decode-metrics` antes de
emitir el flag: si la versión instalada no lo soporta, se omite y el server arranca igual.

## 6. Aplicar un cambio

```bash
uv run llm-server config-migrate && uv run llm-server update
systemctl --user restart llm-server-manager llm-server-router
```

Un cambio de flag de servidor cuesta **una recarga** de modelo. Nunca lo apliques escribiendo
`config.yaml` en caliente.

## 7. Errores ya cometidos

- Comparar vLLM contra buun con distinto `max_tokens` o sin `thinking` → conclusión inválida.
- Aplicar `batch/ubatch/keep` a vLLM (no existen).
- Declarar que "faltan parches" sin mirar `patches-vllm/README.md`.
- Olvidar reaplicar parches tras upgrade y luego depurar `timings` ausentes.
