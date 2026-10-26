# Parches vLLM (Qwen3.5 / Qwen3.8 sobre vLLM 0.32)

Inventario de los parches que aplicamos al wheel de vLLM del venv del gateway, de
dónde salen y cómo volver a aplicarlos tras un `uv sync` / reinstalación / subida
de versión. El ejecutor es `tools/apply_vllm_patch.sh`:

```bash
tools/apply_vllm_patch.sh            # aplica todos los *.patch de este directorio
tools/apply_vllm_patch.sh --revert   # revierte los que estén aplicados
```

El script detecta el `site-packages` del venv
(`$LLM_SERVER_VENV`, por defecto `~/.local/opt/llm-server/venv`), prueba el
parche con `--dry-run` (idempotente) y guarda un `.bak-<timestamp>` de cada
fichero tocado. **Re-ejecutarlo tras cualquier upgrade de vLLM.**

> Nuestro venv es **vLLM 0.32.0**. La receta de referencia
> ([xvus/syv-ai-qwen38-27b-rtx3090](https://github.com/xvus/syv-ai-qwen38-27b-rtx3090))
> está clavada en **0.28.0**: sus parches pueden no aplicar limpios y necesitan
> rebase. Los cuatro de abajo son 0.32-específicos.

## Aplicados (este repo)

| Parche | Fichero(s) | Qué hace |
|---|---|---|
| `vllm-0.32-qwen35-embed-quant.patch` | `model_executor/models/qwen3_5.py` | Respeta el `quant_config` del embedding (checkpoint AutoRound trae `embed_tokens` cuantizado; sin esto vLLM construye BF16 y falla al cargar). |
| `vllm-0.32-qwen35-mtp-share-heads.patch` | `model_executor/models/qwen3_5_mtp.py` | El módulo MTP comparte `lm_head` con el target; sin esto el drafter no ve los pesos y la aceptación se hunde. |
| `vllm-0.32-response-timings.patch` | `entrypoints/openai/chat_completion/{protocol,serving}.py` | Añade el bloque `timings` (`prompt_n`, `prompt_per_second`, `predicted_n`, `predicted_per_second`) a la respuesta OpenAI, como llama.cpp. Lo leen `tools/bench_chat_timings.py` y llama-swap. |
| `vllm-0.32-slot-print-timing.patch` | `v1/engine/output_processor.py` | Imprime el tiempo por slot en el log del motor (diagnóstico). |

## Pendientes de la receta 3090 (con procedencia)

Fuente: <https://github.com/xvus/syv-ai-qwen38-27b-rtx3090/tree/main/patches>
(rama `main`, vLLM 0.28.0). Prioridad por impacto en decode C1 según su README
y `docs/optimizations.md`.

| Prioridad | Parche (receta) | Qué aporta | Notas de port a 0.32 |
|---|---|---|---|
| 1 | `spec-decode-attn.patch` | Split-KV attention para el verify multi-query (5 queries): FA2 deja 58 de 82 SMs ociosos. Con `VLLM_SPEC_DECODE_ATTN=1`; solo KV bf16. | Toca el path de spec-decode attention; validar contra el backend FLASH_ATTN de 0.32. |
| 2 | `sampler-small-topk-fast-softmax.patch` | Top-k sin sort + softmax multi-bloque; drafts truncados al top-k/top-p del target. | Sampler reescrito en 0.32; probable rebase manual. |
| 3 | `qwen3_5-mtp-draft-vocab.patch` | Vocabulario de draft recortado (40k) para el MTP. Con `prepare/build_draft_vocab.py` (ids contados sobre outputs del propio modelo: 97,5 % de cobertura vs 92 % de un vocab web). | Depende de que el MTP acepte `draft_vocab_size`/ids; comprobar firma en 0.32. |
| 4 | `speed-knobs-envs.patch` | Env knobs de la receta (`VLLM_SPEC_DECODE_ATTN`, `VLLM_PREFILL_ATTN`, ...). | Cosmético si no se usan esos envs. |
| 5 | `hybrid-kv-groups-v2-cudagraph.patch` | Evita que las 5 capas sliding-window del drafter padeen las capas attention/GDN del target (78 vs 105 KB/token de pool). | Solo relevante con drafter externo; verificar con `--mamba-cache-mode align`. |
| 6 | `marlin-int8-layer-select.patch`, `marlin-int8-negative-scales.patch`, `marlin-tune-table.patch`, `marlin-repack-staged-sm80.patch` | W4A8: activaciones int8 por token sobre Marlin → +29 % prefill. Env `VLLM_MARLIN_INPUT_DTYPE=int8`, `VLLM_MARLIN_INT8_INCLUDE_RE=mlp\|linear_attn\|self_attn`. | Parches de kernels; alto riesgo de conflicto con 0.32. |
| 7 | `triton-prefill-attn-int8.patch` | Atención prefill int8-QK para hd256 (1,27-1,35× en atención, ~+5 % e2e a 51k). | Companion de W4A8; no standalone. |
| 8 | `int4-kv-per-token-head.patch` | KV int4/int8 per-token-head (perfil CTX=long/huge de la receta). | Alternativa a `fp8` cuando se quiere más contexto. |
| 9 | `xgrammar-spec-terminated.patch` | Cerrado de gramáticas con spec-decode (tool calls). | Toca xgrammar; revisar versión. |
| 10 | `vision-tower-cpu-offload.patch` | Descarga el vision tower a CPU cuando no se usa (ahorra ~0,9 GiB de VRAM). | Solo si volvemos a `--language-model-only`. |
| — | `dflash2-*.patch` (7) + `prepare/fetch_dflash2.py` | Drafter DFlash2: 7 drafts en un pase, 127-131 tok/s C1, 381 citando contexto. | **No en 0.32**: la receta lo corre nativo en 0.28.0. Requiere port a V2 runner de 0.32. |

## Pipeline de prepare de la receta (pendiente)

Fuente: <https://github.com/xvus/syv-ai-qwen38-27b-rtx3090/tree/main/prepare>

| Script | Qué genera |
|---|---|
| `build_draft_vocab.py` + `draft_vocab_ids.json` | Vocab de draft 40k desde outputs del modelo (97,5 % cobertura). |
| `quant_lm_head.py` | `lm_head` int4 GPTQ calibrado (850→215 MB; 1,27→0,65 GB por verify). |
| `quant_mtp.py` | Módulo MTP int4 GPTQ. |
| `quant_embed.py`, `quant_heads_stream.py` | Embedding y cabezas por streaming. |
| `fetch_dflash2.py`, `fetch_fast_variant.py` | Drafter DFlash2 / variante "fast" pre-cuantizada. |

## Knobs de config ya aplicados

En `catalog.json` → `server_overrides.vllm` (se renderizan a `vllm-server`):

| Knob | Valor | Por qué |
|---|---|---|
| `speculative_config.draft_sample_method` | `probabilistic` | Drafts muestreados, no argmax: sube aceptación con `temperature > 0` (la receta mide 2,66→3,23 tok/paso en dflash2). A `temperature 0` es idéntico a `greedy`. |
| `mamba_cache_mode` | `align` | Modelo híbrido (GDN) + `enable_prefix_caching`: alinea el estado recurrente al bloque de prefijo. |
| `enable_per_request_metrics` | `true` | Emite `timings` por petición (los lee el bench y llama-swap). |
| `enable_force_include_usage` | `true` | `usage` en respuestas en streaming. |

Ya estaban: `kv_cache_dtype fp8`, MTP `num_speculative_tokens 3`, `enable_prefix_caching`,
`max_cudagraph_capture_size 16`, `max_num_seqs 2`, `max_num_batched_tokens 1024`,
`limit_mm_per_prompt {image:2,video:0}` (visión).

## Medición (2026-10-26, 2×4090, TP=2, 1,1k prompt, 256 gen, `temperature 0`)

| Config | decode mediana (5 reps) | prefill |
|---|---|---|
| `greedy` + `align` | 107,5 tok/s | 1930 |
| `probabilistic` + `align` | 107,1 tok/s | 1930 |
| `probabilistic` sin `align` | 107,3 tok/s | 1930 |

- A `temperature 0` los drafts argmax y muestreados coinciden → **sin diferencia**
  (la ganancia de `probabilistic` aparece con temperatura > 0).
- `align` no cambia decode; su valor es el reuse de estado recurrente en
  peticiones con prefijo compartido.
- Techo de referencia de la receta en una 3090: 121 tok/s C1 con drafter int4 +
  vocab de draft + split-KV verify; esos tres son los pendientes 1-3.
