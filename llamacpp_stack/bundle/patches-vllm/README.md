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
| `vllm-0.32-dflash2-lookup.patch` | `v1/worker/gpu/spec_decode/dflash2/{speculator,lookup}.py` + `v1/worker/gpu/model_runner.py` | **[port 0.32]** Lookup-augmented drafting (LABD): propone la continuación de una aparición previa del sufijo actual (copia exacta del contexto) y reescribe esos logits a masa puntual (rechazo exacto). `VLLM_DFLASH2_LOOKUP=0` lo desactiva. |
| `vllm-0.32-qwen35-mtp-draft-vocab.patch` | `model_executor/models/qwen3_5_mtp.py` | **[port 0.32]** Head de draft truncado a 40 960 ids (`mtp.draft_lm_head.*` + `mtp_draft_vocab_ids.pt`); el resto de logits va a `-inf`. El rechazo sigue siendo exacto. Fuente: `patches/qwen3_5-mtp-draft-vocab.patch` de la receta. |
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
| ~~3~~ | ~~`qwen3_5-mtp-draft-vocab.patch`~~ ✅ **aplicado** | Vocab de draft 40 960 ids (ver abajo). | Portado a 0.32 (ver fila en Aplicados). |
| 4 | `speed-knobs-envs.patch` | Env knobs de la receta (`VLLM_SPEC_DECODE_ATTN`, `VLLM_PREFILL_ATTN`, ...). | Cosmético si no se usan esos envs. |
| 5 | `hybrid-kv-groups-v2-cudagraph.patch` | Evita que las 5 capas sliding-window del drafter padeen las capas attention/GDN del target (78 vs 105 KB/token de pool). | Solo relevante con drafter externo; verificar con `--mamba-cache-mode align`. |
| 6 | `marlin-int8-layer-select.patch`, `marlin-int8-negative-scales.patch`, `marlin-tune-table.patch`, `marlin-repack-staged-sm80.patch` | W4A8: activaciones int8 por token sobre Marlin → +29 % prefill. Env `VLLM_MARLIN_INPUT_DTYPE=int8`, `VLLM_MARLIN_INT8_INCLUDE_RE=mlp\|linear_attn\|self_attn`. | Parches de kernels; alto riesgo de conflicto con 0.32. |
| 7 | `triton-prefill-attn-int8.patch` | Atención prefill int8-QK para hd256 (1,27-1,35× en atención, ~+5 % e2e a 51k). | Companion de W4A8; no standalone. |
| 8 | `int4-kv-per-token-head.patch` | KV int4/int8 per-token-head (perfil CTX=long/huge de la receta). | Alternativa a `fp8` cuando se quiere más contexto. |
| 9 | `xgrammar-spec-terminated.patch` | Cerrado de gramáticas con spec-decode (tool calls). | Toca xgrammar; revisar versión. |
| 10 | `vision-tower-cpu-offload.patch` | Descarga el vision tower a CPU cuando no se usa (ahorra ~0,9 GiB de VRAM). | Solo si volvemos a `--language-model-only`. |
| ~~—~~ | ~~`dflash2-*.patch`~~ ✅ **nativo en 0.32** (lookup ✅ portado; ngram-chains/adaptive-emitted pendientes) | Drafter DFlash2: 7 drafts en un pase. Medido aquí: **127,1 tok/s C1** (vs 116,5 con MTP) y 132,6 en tarea de citar a 27k. | vLLM 0.32 trae `qwen3_dflash2.py` + `DFlashProposer`; solo hay que bajar el drafter. Los parches `dflash2-*.patch` de la receta (lookup, ngram chains) siguen sin portar: sin ellos no se llega a sus 381 en citas largas. |

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

## Draft vocab aplicado (2026-10-26)

La vía normal es el propio gateway:

```bash
llm-server vllm-draft-vocab --model 'Qwen3.8-27B-AutoRound-INT4-[vLLM]'   # uno
llm-server vllm-draft-vocab                                              # todos los vLLM con MTP
llm-server vllm-draft-vocab --force                                      # regenerar
```

Se ejecuta **automáticamente al hacer `llm-server add` de un modelo vLLM con pesos
`mtp.*`** (best-effort; `LLM_SERVER_DRAFT_VOCAB=0` lo desactiva). También se puede
hacer a mano:

```bash
# 1. backup del índice y de los extras del modelo (el script también hace su .bak-draft)
cp -a "$MODEL/model.safetensors.index.json" "$MODEL/model.safetensors.index.json.bak-draftvocab"
cp -a "$MODEL/model_extra_tensors.safetensors" "$MODEL/model_extra_tensors.safetensors.bak-draftvocab"
# 2. construir el head truncado (213 MB en este checkpoint int8: 40 960 x 1 280 int32)
python tools/build_draft_vocab.py "$MODEL" --ids tools/draft_vocab_ids.json
# 3. reiniciar el router (recarga vLLM); el log debe decir:
#    "MTP drafter uses a 40960-token draft head"
```

- `tools/build_draft_vocab.py` y `tools/draft_vocab_ids.json` son copias vendorizadas
  de `prepare/` de la receta (ids contados sobre danés/inglés/código + outputs;
  cobertura held-out ~95 %). Para dominios propios, regenerar con `--n 40960 --corpus f1 f2 ...`.
- Aviso benigno en el log: las escalas del head se guardan fp16 y vLLM las carga en
  bf16 (`Attempted to load weight draft_lm_head.weight_scale with dtype torch.float16
  into parameter with dtype torch.bfloat16`); los valores son correctos.
- Desactivar: `MTP_DRAFT_VOCAB=0` en el entorno del motor.

## DFlash2 en vLLM 0.32 (aplicado 2026-10-26)

**Sí hay DFlash2 en 0.32**: el wheel trae `vllm/v1/spec_decode/dflash.py`
(`DFlashProposer`), `vllm/model_executor/models/qwen3_dflash2.py`
(`DFlash2Qwen3ForCausalLM`, registrado como `DFlash2DraftModel`) y el speculator
`dflash2` en `transformers_utils/configs/speculators/algos.py`. No hace falta
portar el runner: el drafter pre-cuantizado de la receta carga tal cual.

```bash
# drafter W4A16-GPTQ (1.2 GB; el bf16 original son 3.85 GB)
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("syvai/Qwen3.8-27B-DFlash2-W4A16",
                  local_dir="/var/llamacpp_models/syvai/Qwen3.8-27B-DFlash2-W4A16",
                  allow_patterns=["*.json", "*.safetensors", "README.md"])
EOF
```

`catalog.json` → `server_overrides.vllm.speculative_config`:

```json
{
  "method": "dflash",
  "model": "/var/llamacpp_models/syvai/Qwen3.8-27B-DFlash2-W4A16",
  "num_speculative_tokens": 7,
  "draft_sample_method": "probabilistic"
}
```

- `draft_sample_method: probabilistic` es obligatorio aquí: con el default
  (`greedy`) la aceptación se hunde (la receta mide 2,66 vs 3,23 tok/paso).
- Aviso benigno: `no KV cache group could be identified as the draft model's`
  (DFlash2 consume los hidden states del target, no tiene KV propio).
- Revertir a MTP: `{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}`
  (el draft vocab 40 960 sigue en el modelo y se reutiliza).

### Lookup drafting (LABD) — aplicado

Portado de `patches/dflash2-lookup-drafting.patch` (1.073 líneas) al speculator de
0.32: `suffix_lookup` (encuentra la ocurrencia más reciente del sufijo en el
historial del request), `fuse_draft` (funde la propuesta con la del drafter; cabeza
con umbral `nstrong`/acuerdo y cola libre) y `_point_mass_draft_logits` (q = masa
puntual, rechazo exacto). Buffers y llamada en `DFlash2Speculator`, `set_req_states`
desde el model runner.

Knobs (env del motor): `VLLM_DFLASH2_LOOKUP=0` off; `..._NMIN` (6), `..._NSTRONG` (8),
`..._AGREE` (2), `..._NMAX` (32).

| Prueba | sin lookup | con lookup |
|---|---|---|
| Copiar un pasaje corto (142 prompt) | — | **205,1 tok/s, salida idéntica** |
| Citar 27k (`--task quote`) | 132,6 | **193,6-201,6** (+46-52 %) |
| Resumen C1 (prosa) | 117,7-154,8 (med 127,1) | 108,5-141,2 (med 115,8) |

Coste en prosa: 0-9 % (su receta reporta 3-20 % en contexto largo mixto).

### Bloque 15 (modo copia, opt-in)

El port incluye el desacople `draft_block` (7, lo que el checkpoint sabe hacer) del
bloque de verify, así que `num_speculative_tokens: 15` funciona: el drafter emite 7
y el lookup rellena 7..14.

| | k=7 (default) | k=15 |
|---|---|---|
| Citar 27k | 193-202 | **259** (+28-34 %) |
| Prosa C1 | 105-127 | **94** (-19 %) |

El coste en prosa es fijo porque no portamos el controlador adaptativo de la receta
(`next_num_draft_tokens`, decide bloque largo solo mientras la copia está en marcha):
sin él, se verifican 16 tokens en cada paso. Por eso el default sigue en **k=7**.
Para activar el modo copia en `catalog.json` → `server_overrides.vllm`:

```json
"speculative_config": {"method": "dflash", "model": ".../DFlash2-W4A16",
                       "num_speculative_tokens": 15, "draft_sample_method": "probabilistic"},
"max_cudagraph_capture_size": 64
```

y `llm-server update` (recarga el modelo).

### Medición

| | MTP d3 + vocab | DFlash2 d7 |
|---|---|---|
| C1 (5 reps, 1,1k/256) | 116,5 | **127,1** (117,7-154,8) |
| prefill | ~1975 | ~1900 |
| C4 agregado | 185,9 | 184,8 |
| C8 agregado | 246,6 | 244,9 |
| citar 27k | — | **132,6** decode, 2033 prefill |
| KV pool | 644 376 tok (2,46× a 262k) | 448 447 tok (1,71×) |
| VRAM | 22,1 GiB/GPU | 21,9 GiB/GPU |

Coste: el pool de KV baja ~30 % (1,71× en vez de 2,46× a 262 144), así que con 8
slots caben menos peticiones *largas* a la vez. A cambio, +9 % en C1.

## Concurrencia (2026-10-26)

`max_num_seqs` probado a 2 / 4 / 8 con 1,1k de prompt y 256 generados (agregado =
tokens / pared, incluye prefill; per-request desde `timings`):

| max_num_seqs | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| 2 | 120,4 | — (cola) | — | — |
| 4 | 111-115 | 121-130 | **181,8** | 192,9 (cola) |
| **8** | **116,5** | **124,9** | **185,9** | **246,6** |

- C1 baja ~3 % al pasar de 2 a 8 (dentro del ruido de la receta, 122±5), pero el
  agregado a C8 sube de ~193 a **247 tok/s**. Se queda en **8**.
- El pool queda en **644 376 tokens** de KV (2,46× a 262 144 por petición):
  caben 8 peticiones cortas, no 8 de 262k (esas hacen cola/preempt).
- VRAM: 22,1 GiB de 24,5 por GPU con el modelo cargado.

## Medición (2026-10-26, 2×4090, TP=2, 1,1k prompt, 256 gen, `temperature 0`)

| Config | decode mediana (5 reps) | prefill |
|---|---|---|
| `greedy` + `align` | 107,5 tok/s | 1930 |
| `probabilistic` + `align` | 107,1 tok/s | 1930 |
| `probabilistic` sin `align` | 107,3 tok/s | 1930 |
| `probabilistic` + `align` + **draft vocab 40 960** | **120,4 tok/s** | 1960 |

- A `temperature 0` los drafts argmax y muestreados coinciden → **sin diferencia**
  (la ganancia de `probabilistic` aparece con temperatura > 0).
- `align` no cambia decode; su valor es el reuse de estado recurrente en
  peticiones con prefijo compartido.
- Techo de referencia de la receta en una 3090: 121 tok/s C1 con drafter int4 +
  vocab de draft + split-KV verify; esos tres son los pendientes 1-3.
