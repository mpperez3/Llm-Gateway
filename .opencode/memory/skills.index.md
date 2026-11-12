---
memory_file: skills_index
version: 1
updated: 2026-11-12
load: always
scope: repo
---
# Índice de skills (siempre visible)

Se inyecta en cada sesión porque el índice de skills nativo se elimina cuando `triage` está
activo. **Si la tarea cae en una de estas filas, lee el `SKILL.md` completo antes de actuar.**

| Skill | Dispara con |
|---|---|
| `llm-server-ops-triage` | 502, 500, 429, 503, stream cortado, modelo que no carga, latencia rara |
| `llm-server-remote-deploy` | desplegar, sincronizar, "en remoto", 192.168.110.41, mpperez3 |
| `llm-server-perf-bench` | tok/s, comparar configs, regresión, "¿va más rápido?", medir |
| `llm-server-model-tuning` | subir tok/s, "que quepa", ctx, batch, tensor_split, cache_quant, parallel |
| `llm-server-autotune` | modelo nuevo, GGUF/safetensors recién añadido, `auto_ctx_failed`, OOM al cargar |
| `llm-server-installer-update` | install, update, migrar user↔system, desinstalar, "ha quedado a medias" |
| `llm-server-vllm-backend` | vLLM, `[vLLM]`, parches, `timings` ausentes, draft vocab, reasoning parity |
| `llm-server-mmproj-vision` | imágenes, visión, mmproj, lazy, "no procesa imágenes" |
| `llm-server-kv-cache-diagnosis` | prefill espurio, KV cache, "tarda mucho en conversaciones largas" |
| `llm-server-request-forensics` | "guarda esta request", captura #N, respuesta malformada, fixture |
| `llamabench-config-advisor` | "config óptima", mejor quant, recomendación de parámetros |

## Reglas de uso

1. Una skill se lee **completa**; no se improvisa su procedimiento.
2. Si dos aplican (p. ej. tuning + bench), el orden es: `perf-bench` para medir, `model-tuning`
   para cambiar.
3. Si la tarea no encaja en ninguna, no fuerces una skill: trabaja y, si el procedimiento se
   repite por tercera vez, propone una nueva.
4. Toda skill nueva se añade aquí, en `AGENTS.md §6` y en `.opencode/skills/`; el test
   `tests/test_project_memory.py` falla si las tres listas no coinciden.
