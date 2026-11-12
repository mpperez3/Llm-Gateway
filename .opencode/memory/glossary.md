---
memory_file: glossary
version: 1
updated: 2026-11-12
load: always
scope: repo
---
# Glosario del proyecto

Vocabulario propio de este repo y del usuario. **Sirve para no inventar componentes ni
confundir motores.** Cada término lleva dónde se verifica; si el código cambia, actualiza la
entrada. Ante duda, gana el código.

## Producto y servicios

| Término | Significado | Verificado en |
|---|---|---|
| **llm-server** | Nombre público del producto y del CLI. `heimdall-gateway` es alias deprecado | `pyproject.toml`, `AGENTS.md §1` |
| **gateway** (`:11435`) | Proceso que aplica la política: dedup, budget de reasoning, réplicas, tool-repair. Escribe `api-requests.log`. Unit `llm-server-manager` | `llamacpp_stack/cli/gateway.py` |
| **router** (`:11436`) | Punto de entrada de `llama-swap`. Unit `llm-server-router` | `install.py` (render de la unit) |
| **guard** / `llama-swap-guard` | Proceso que sirve el puerto público del router y lanza/proxya `llama-swap`. **Sí existe**: es un subcomando del CLI (`llama-swap-guard`) que la unit `llm-server-router` ejecuta con `--llamaswap-bin … --listen-port 11436` | `llamacpp_stack/cli/parser.py` (`"llama-swap-guard"`), `install.py:4492` |
| **llama-swap** | Supervisor/router de terceros que arranca procesos de modelo y expone `/v1/models`, `/upstream/<model>/`, `/api/metrics/activity` | `docs/LLM_INSTALL.md` |
| **dos entry points** | `:11435` aplica política; `:11436` no. Un fix solo en `:11435` no llega a Hermes/web UI | `decisions.md` D10 |

## Motores (cuatro opcionales distintos — no confundirlos)

Etiquetas reales del instalador (`install.py`, `OPTIONAL_ENGINE_LABELS`):

| Etiqueta | Nombre interno | Qué es | Binario |
|---|---|---|---|
| `beellama (kvarn4/MTP)` | `beellama` | Build de llama.cpp con KV cache **KVarN** y MTP | `llama-server*` |
| `vLLM` | `vllm` | Backend Hugging Face nativo (beta) | `vllm-server` |
| `EXL3 (ExLlamaV2+DFlash2)` | `exllama` | Backend EXL3 **legacy**, en retirada | `llama-server-exllama` |
| `buun (EXL3+FP8 native)` | `buun` | Fork `buun-llama-cpp`: EXL3 y FP8 nativos | `llama-server-buun` |

**No son sinónimos.** `beellama` ≠ `buun`. `exllama` y `buun` cargan EXL3, pero por caminos
distintos y con flags distintas.

## Modelo, rutas y placement

| Término | Significado |
|---|---|
| **catálogo** | `catalog.json`: inventario de modelos y `server_overrides` por modelo |
| **settings** | `conf.json`: defaults globales/familia, réplicas, logging, experimental |
| **runtime** | `config.yaml`: **generado**, nunca se edita a mano ni en caliente |
| **ruta** | Entrada de modelo en `config.yaml`, con su `cmd` ya renderizado |
| **variante de ruta** | `<model_id>__gpu0`, `<model_id>__gpu1`…: el mismo modelo publicado por GPU para elegir placement sin reescribir config |
| **base** | Instancia principal de un modelo (id público) |
| **réplica** | Instancia interna `<model_id>__replica_N`; el cliente sigue usando el id público |
| **afinidad** | Vínculo conversación→instancia. Toda escritura pasa por `bind()`; hay `min_dwell_s`, `max_transfers`, `evacuate_cooldown_s` |
| **warmer** | `experimental.idle_gpu_warmer`: carga una ruta barata en una GPU ociosa |
| **eviction** | Desalojo de una instancia para cargar otra; nunca se interrumpe una con requests en vuelo |

## Estados y errores

| Señal | Significado |
|---|---|
| `429 overloaded` | Límite `LLM_SERVER_MAX_CONCURRENT_PER_MODEL`; no es fallo del motor |
| `503 model_loading` | `claim_loading`: otro load tiene reclamada la GPU; el primero gana |
| `model_request_blocked_overloaded` | Evento del log al rechazar por concurrencia |
| `bundle_ref` / **crash bundle** | Referencia al paquete de diagnóstico que se genera en un 502 (journal + `nvidia-smi`) |
| **raw ring** | `api-raw-requests.log`: últimas 10 requests sin procesar (1 MiB, fallback `/tmp`) |
| `slot print_timing` | Línea de journal que **solo** emite `llama-server`; en EXL3/buun/vLLM no existe |

## Rendimiento y cuantización

| Término | Significado |
|---|---|
| **`ctx_size`** | Contexto total del motor. Con `parallel=N`, cada slot recibe `ctx_size / N` |
| **`parallel`** | Slots concurrentes del mismo proceso; multiplica el KV cache |
| **`tensor_split`** | Reparto entre GPUs, normalizado; **nunca < 1** por GPU (offload a CPU → ~0,27 t/s) |
| **KVarN / kvarn4 / kvarn8** | Familia de cuantización de KV cache del build beellama; `kvarn8` a 262K ya provocó OOM |
| **turbo4 / turbo8** | Codecs de KV cache usados en buun (`cache_type_k/v`, `cache_quant`) |
| **`grid_size` (`-gs`)** | Tiling por GPU en EXL3/buun; palanca real de prefill |
| **MTP** | Multi-token prediction: drafter integrado en el modelo; opción por defecto en este proyecto |
| **DFlash2** | Drafter especulativo alternativo. Con `ctx_size_draft: 0` hereda el ctx del target → ~5 GiB de KV → OOM a 262K en 24 GiB |
| **`mmproj`** | Proyector de visión. `mmproj_mode: lazy` lo carga bajo demanda con `vision_sticky_ttl_s` |
| **`api_ctx_factor`** | Factor (default 0.5) con el que el gateway deriva el contexto publicado a clientes |
| **`reasoning_budget`** | `half_context` = `min(half_ctx, gen_lim − min(1024, gen_lim/8))`, con `gen_lim = min(max_tokens, predict)` |

## Dedup

| Término | Significado |
|---|---|
| **singleflight** | Una request idéntica se procesa una vez; las demás esperan al líder |
| **líder / seguidor** | La primera request y las idénticas que reciben su resultado |
| **`principal_hash`** | `sha256(api_key)`; nunca se comparten resultados entre principales distintos |
| **tee buffer** | Buffer del líder para servir al seguidor (`tee_buffer_lines`, `tee_buffer_bytes`); si se llena, el seguidor recibe drop, no frena al líder |
| **TTL 600 s** | Caché de resultados exitosos; solo `2xx`, solo `POST`, solo `:11435` |

## Alias del usuario

| El usuario dice | Significa |
|---|---|
| "41", "el otro equipo", "remoto" | `mpperez3@192.168.110.41` (`sing-mgpu04`) |
| "local", "aquí" | esta máquina de desarrollo (2× RTX 4090 24 GB) |
| "beellama" | build `beellama` (kvarn4/MTP), **no** buun |
| "el guard" | `llama-swap-guard` en `:11436` |
| "el histórico" | `configs/history/` |
| "la conf" | normalmente `catalog.json` (por modelo) o `conf.json` (global) — preguntar si es ambiguo |
| "full de contexto" / "máximo" | 263–264K por slot, sin excepciones |
