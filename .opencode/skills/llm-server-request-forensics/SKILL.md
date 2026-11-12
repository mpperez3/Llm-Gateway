---
name: llm-server-request-forensics
description: "Captura, saneado y replay de peticiones/respuestas reales para convertir un bug en un fixture de regresión. Úsala SIEMPRE que el usuario diga \"guarda esta request\", \"captura #N\", \"esta respuesta salió mal\", \"mira la última petición\", \"guárdalo como ejemplo para tests\", o cuando haya que investigar una respuesta malformada, cortada o en otro idioma. Evita inventar peticiones de prueba: aquí está cómo extraer las reales del log."
---

# Forense de peticiones

El usuario ha pedido esto muchas veces y el patrón es siempre el mismo: **hay una petición real
que produjo el bug; hay que capturarla, sanearla, guardarla y convertirla en test.**

## 1. Localizar la petición exacta

```bash
ls -lt ~/.local/state/llm-server/api-requests.log* | head -5   # el activo es el de mtime mayor
uv run llm-server requests --lines 200
cat ~/.local/state/llm-server/api-raw-requests.log | tail -c 2000   # últimas 10 sin procesar
```

- El symlink `api-requests.log` **puede apuntar al fichero antiguo** tras un split; usa el
  `.partN` con mtime más reciente.
- `api-raw-requests.log` es el cuerpo crudo, tal cual llegó. Es la fuente preferida.
- El log rotado tiene la respuesta, el `status`, `bundle_ref` y eventos
  (`dedup_hit`, `tool_repair`, `model_load_*`).

Si el usuario da una referencia (`Capture #57/v1/chat/completions`), busca ese identificador
en el log antes de preguntar nada.

## 2. Saneado obligatorio antes de guardar

Nunca commitees una petición real sin limpiar:

- `Authorization`, `X-Api-Key`, cookies, `api_key`.
- Rutas privadas del usuario, nombres de modelos internos si el fixture va a ser público.
- Contenido personal del usuario (sustituir por texto neutro preservando **forma y tamaño**).

Lo que **no** se toca: estructura de `messages`, longitud de cada mensaje, `tools`, orden de
roles, `max_tokens`, `stop`, `temperature`. El bug depende de eso.

El repo ya tiene ejemplos de referencia:
`tests/fixtures/gateway_raw_examples/extra_long_qwen_481_raw.json` (+ `_headers.txt`,
`_meta.json`) y `tests/fixtures/regressions/`.

## 3. Guardar con metadatos

```text
tests/fixtures/regressions/<sintoma>-<YYYYMMDDTHHMMSS>.json
```

El JSON debe incluir un bloque `_meta` con:

```json
{"_meta": {"model": "...", "backend": "buun|llama.cpp|vllm|exllama", "ctx_size": 0,
           "parallel": 0, "captured_at": "2026-11-02T12:00:00Z",
           "symptom": "respuesta en chino / useruser / corte al primer token",
           "source": "api-raw-requests.log", "sanitized": true}}
```

Sin `_meta` el fixture es inútil dentro de un mes.

## 4. Replay

- **Offline (sin GPU, preferido):** los splitters y la lógica de reparación se pueden alimentar
  con el fixture. Ver `tests/test_beellama_exllama_parity.py`: replay de `StreamSplitter` sin
  modelo, con fixtures saneados.
- **Online:** relanzar la petición por el camino del cliente (`:11435`/`:11436`), no por el
  puerto del backend.
- Compara **antes/después** con la misma petición y el mismo modelo. Si cambias dos cosas, no
  sabes qué arregló el bug.

## 5. Convertirlo en test

1. Añade el caso a un test existente o crea `<sintoma>_test.py`.
2. El test debe **fallar con el código antiguo** (caracterización) y pasar con el nuevo.
3. Nada de GPU ni de modelo real si el bug está en el gateway/splitter.
4. Sanity: el fixture no contiene secretos (`grep -i "authorization\|api.key\|bearer"`).

## 6. Cuándo no usar esta skill

- El bug es de rendimiento y no de contenido → usa `llm-server-perf-bench`.
- El bug es un 502/OOM/latencia de carga → usa `llm-server-ops-triage`.

## 7. Salida esperada

`fixture saneado` + `ruta` + `comando de replay` + `test que lo cubre`. Si no puedes reproducir
el bug con la captura, **dilo**: significa que falta contexto (modelo, kv cache o config) y
seguir tocando código a ciegas es peor.
