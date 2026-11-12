---
name: llm-server-perf-bench
description: Cómo medir rendimiento (tok/s) en LLM Server sin autoengañarse. Úsala SIEMPRE que se comparen configuraciones (spec decode, KV dtype, ctx, flags de vLLM/buun), se investigue una regresión de velocidad, se valide una "mejora" de kernel/receta o se decida un cambio de catálogo por rendimiento. Contiene el protocolo que evitó conclusiones invertidas (medir 512 tokens dio lo contrario que 1024), las herramientas del repo y los números de referencia medidos en esta máquina.
---

# Benchmarking de LLM Server

Una comparación mal montada cambia la configuración de producción en la dirección
equivocada. Esta skill existe por un caso real: se concluyó "quitar speculative
decoding es un 22 % más rápido" midiendo prosa sintética, 512 tokens, `thinking`
apagado y el puerto del backend. Con los prompts reales del proyecto y 1024 tokens,
lo correcto era lo contrario: **DFlash 7 era un 70 % más rápido** a C1 y un 39 % a
164k de contexto.

## Reglas duras

1. **Nunca decidas con menos de 1024 tokens de salida.** La aceptación del drafter
   mejora conforme la respuesta se asienta; 256-512 tokens sobreponderan el coste
   del draft y dan la conclusión opuesta.
2. **`thinking` activado (el defecto del servidor) salvo que la pregunta sea sobre
   apagarlo.** El razonamiento es el texto más predecible: apagarlo destruye la
   aceptación y hunde artificialmente a spec decode.
3. **Prompts reales del proyecto, no relleno.** Usa
   `tests/fixtures/gateway_raw_examples/extra_long_qwen_481_raw.json` (petición real
   de 481 items) y los 8 prompts internos de la receta de
   `tools/bench_real_prompts.py --workload recipe`. El relleno tipo
   "línea 000001: el rápido zorro..." es degenerado y sesga todo.
4. **Mide por el camino del cliente** (`http://127.0.0.1:11436` o `:11435`), no por
   el puerto del backend (`:18080`). El cliente real atraviesa el guard.
5. **Siempre corre el control en la misma sesión.** Spec vs no-spec, con el mismo
   prompt, el mismo `max_tokens` y el mismo `thinking`. La aceptación depende del
   contenido: un número aislado no significa nada.
6. **Un cambio de variable por recarga.** Cambiar dos cosas a la vez invalida la
   comparación.
7. **Dos repeticiones como mínimo, y reporta la mediana.** La dispersión entre
   corridas es del 3-8 %.
8. **Contrasta con la receta.** Referencia en RTX 3090 @250 W: 46 tok/s sin spec,
   118 MTP, 121-133 DFlash2 a C1. Si tu medida contradice la receta por 2×,
   sospecha de tu banco, no de la receta.
9. **Nunca cambies producción desde un microbenchmark.** Primero el experimento,
   luego la propuesta, luego el cambio con su recarga y su verificación.
10. **Verifica el argv renderizado** tras cada cambio de catálogo
    (`tr '\0' '\n' < /proc/<pid>/cmdline`), y anota el pool KV
    (`GPU KV cache size`) porque cambia con la configuración.

## Protocolo

```bash
# 0. Aislar de clientes reales antes de medir
uv run llm-server maintenance on     # solo requests locales
# … al terminar: uv run llm-server maintenance off

# 1. Medir A/B con los prompts reales (esta es la herramienta de la skill)
uv run python tools/bench_real_prompts.py --workload fixture --max-tokens 1024 --repeats 2
uv run python tools/bench_real_prompts.py --workload recipe --max-tokens 1024 --repeats 2

# 2. Cambiar UNA variable en catalog.json, recargar y repetir
#    (spec, kv_cache_dtype, compilation_config, ctx...)
systemctl --user restart llm-server-manager llm-server-router
curl -s -o /dev/null -w '%{http_code}\n' :11436/health   # warmup aparte, no se mide

# 3. Aceptación del motor durante la medida
journalctl --user -u llm-server-router --since "10 min ago" | grep "SpecDecoding metrics"

# 4. Volver a la mejor configuración y verificar
# 5. Guardar snapshot en configs/history/ y entrada en .opencode/memory/events.md
```

## Casos que engañan

| síntoma | causa | qué hacer |
|---|---|---|
| spec "más lento" | prosa libre / `thinking` off / 512 tokens | repetir con prompts reales y 1024 tokens |
| dos corridas idénticas dan 34 s y 52 s | admisión de prefill (una prefill sola vs chunked) | mirar `prompt_per_second`; no comparar lotes |
| tok/s por stream baja a la mitad con 2 clientes | reparto de batch, no regresión | comparar agregado y por stream por separado |
| contexto largo mucho más rápido de lo esperado | hit de prefix cache | usar nonce/prompt único (cold) |
| PIECEWISE "gratis" | en 0.32 sin los parches de la receta cuesta -64 % a 164k | medir siempre en tu stack |
| salida corrupta (sopas de alfabetos) | residuo de captura FULL + prefix cache | canario en `logging.stream_canary`; ver `docs/` |

## Números de referencia de esta máquina (2026-10-28)

2×24 GB, TP=2, fp8 KV, `max_model_len` 262144, prompts reales, 1024 tokens,
vía `:11436`, thinking on:

| config | C1 receta (8 prompts) | 164 943 tok de contexto real |
|---|---|---|
| sin spec | 80,6 | ~65 |
| MTP 4 | 106,2 | 72,9 |
| **DFlash 7** | **137,1** | **89,7** |
| DFlash 7 + PIECEWISE | 126,0 | 31,9 |

Pool KV: 397 776 tokens con drafter, 739 728 sin él (1,52× vs 2,82× a 262k).
