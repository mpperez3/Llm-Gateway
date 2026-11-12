---
name: llm-server-kv-cache-diagnosis
description: Diagnosticar por qué un turno corto reprocesa decenas de miles de tokens de prompt (prefill espurio, cache_n bajo, "restored context checkpoint" muy atrás). Úsala SIEMPRE que se hable de KV cache, prefix cache, prompt processing, "va lento en conversaciones largas", "ha tardado mucho en responder", o cuando alguien proponga tocar checkpoints/cache-ram sin medir. No la uses para latencia de carga del modelo ni para tok/s de decode (usa perf-bench).
---

# Diagnóstico de KV cache

Síntoma de referencia: un turno que añade ~3k tokens reprocesa ~30k de prefill. No es "el
modelo"; es la interacción entre plantilla, `preserve_thinking` y checkpoints en modelos
híbridos. Documento largo: `docs/kv-cache-reuse-hybrid-qwen35.md`.

## 1. Confirmar que existe el problema

```bash
curl -s http://127.0.0.1:11436/api/metrics/activity | jq '.data[] | {model, tokens}'
uv run llm-server requests --lines 50
journalctl --user -u llm-server-router --since "10 min ago" \
  | grep -E "restored context checkpoint|CHECKPOINT_ATTN_ONLY_TRIM|cache_plan_s" | tail -20
```

Señales:

- `restored context checkpoint (pos_min = X, n_tokens = Y)` con `Y` **muy por debajo** del
  tamaño real del historial → se reutiliza poco.
- `cache_plan_s: ... lcp ... reusable ...` → comparar `lcp` con `reusable`.
- `cached_tokens`/`cache_n` a `-1` o 0 en un turno de continuación.

## 2. Las dos causas (comprobar en este orden)

**A. `preserve_thinking=false` + plantilla posicional (Qwen3.5).**
El gateway descarta el bloque de razonamiento anterior; la plantilla cambia prefijos y el LCP
se corta mucho antes del final del historial. *Arreglo:* `preserve_thinking: true` en
`llama_server_family_defaults` del modelo/familia (es una palanca de config, no de código).

**B. Modelo híbrido: el reuse nunca supera el checkpoint exacto más nuevo ≤ LCP.**
No importa que el LCP sea grande si el último checkpoint está atrás. *Palancas:*
`ctx_checkpoints` (16 → 24), `checkpoint_min_step` (1024 → 512), `cache_ram`.

## 3. Medir antes de tocar

Captura dos prompts consecutivos y calcula el LCP real:

```bash
python tools/compare_prompt_captures.py --help
# captura A = turno N, captura B = turno N+1; comparar prefijo común y punto de divergencia
```

Sin esta comparación, "sube los checkpoints" es una apuesta. Puede que el problema sea A y los
checkpoints no cambien nada.

## 4. Aplicar (una palanca por iteración)

```bash
uv run llm-server config-migrate    # añade claves nuevas sin sobrescribir
uv run llm-server update
systemctl --user restart llm-server-manager llm-server-router
```

Tras cada cambio: repetir la medición del §1 con la **misma conversación**. Cambiar modelo,
prompt o `max_tokens` invalida la comparación.

## 5. Qué NO hacer

- Reducir contexto para "que quepa" (invariante 1).
- Bajar la cuantización de KV cache (invariante 2).
- Suponer que EXL3/buun/vLLM tienen prefix cache como llama.cpp: en EXL3 el reuse es limitado y
  `cache_n = -1` es normal; no lo interpretes como bug.
- Activar `logits-all` o depurar con prompts sintéticos: cambia el comportamiento del cache.

## 6. Verificación

Un turno de continuación debe mostrar `cached_tokens`/`cache_n` coherente con el historial y un
`restored context checkpoint` cercano al final. Si sigue reprocesando, revisa la plantilla
(§8 del doc): el problema puede estar en el `chat_template.jinja` del modelo, no en las
palancas.
