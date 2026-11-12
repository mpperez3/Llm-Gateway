---
memory_file: known_issues
version: 1
updated: 2026-11-12
load: on_demand
scope: repo
---
# Trampas conocidas

Errores ya cometidos en este proyecto (por agentes o por el código) con su causa y la acción
correcta. **Léelo antes de depurar cualquiera de estos síntomas.** Si repites uno de estos
errores, es un fallo del agente, no del usuario.

## Errores de agente (los que más veces han pasado)

| Síntoma | Causa | Qué hacer |
|---|---|---|
| El usuario dice "mentira", "no es cierto" | Se afirmó un estado sin comprobarlo (modelo cargado, paralelismo real, VRAM libre) | Verifica con `nvidia-smi`, `curl /v1/models`, `/api/replicas` y una request real antes de afirmar |
| Se cambió la config en vez de medir | Petición de benchmark interpretada como petición de tuning | Si piden comparación, **no toques producción**; mide y presenta tabla |
| Comparación inválida | Distinto quant, distinto `max_tokens`, draft desactivado, prompt sintético | Ver invariantes 16–19 y skill `llm-server-perf-bench` |
| Compilación/reinstalación innecesaria | No se comprobó si el componente ya estaba instalado | `llm-server info`, `--dry-run`, opción skip |
| Se rompió código para desactivar algo | Se ignoró que existía una flag | Buscar la clave en `config-keys --format json` primero |
| Fix aplicado solo en `:11435` | El cliente iba directo a `:11436` | Aplicar en ambos entry points o en el guard |
| Respuestas largas e incomprensibles | Explicación sin conclusión | Conclusión primero, detalle después, y solo si se pide |
| Se paró a medio trabajo | Se pidió confirmación ya dada | Terminar, verificar, reportar |
| Trabajo hecho en el host equivocado | Confusión local vs remoto | Confirmar host antes de tocar nada |
| Se citó un componente inexistente ("el guard") | Se inventó arquitectura | Verificar en `config.yaml`/código antes de nombrar componentes |
| Una skill nunca se dispara | Estaba en `skills/` (raíz) o duplicada en otra ruta | La única ubicación escaneada es `.opencode/skills/<n>/SKILL.md`; las copias viejas van a `.agents/skills/_archive/` |
| Memoria nueva no aparece en `git status` | `/.opencode/` estaba ignorado entero | Ya hay excepción para `.opencode/memory/` y `.opencode/skills/`; no la quites |

## Fallos técnicos recurrentes

| Síntoma | Causa | Qué hacer |
|---|---|---|
| 502 al cargar un modelo | Crash en load: OOM, ctx excesivo, draft incompatible, mmproj | Buscar `bundle_ref` en el JSON 502 y `*_with_bundle` en `api-requests.log.<fecha>[.partN]`; `llm-server logs --lines 200 --journal` |
| 500 `failed to start command 'vllm-server ...'` | El `PATH` de llama-swap no incluye `~/.local/bin` | `tr '\0' '\n' < /proc/$(pgrep -f llama-swap)/environ \| grep ^PATH=`; el wrapper debe exportar `bin_dir` |
| `429 overloaded` | `LLM_SERVER_MAX_CONCURRENT_PER_MODEL` alcanzado | No es un fallo del motor; revisar límite o réplicas |
| `503 model_loading` | Otro modelo cargando en un set de GPUs en conflicto | Esperar; el primer load gana. No forzar |
| `503` con víctima ocupada | El load evictaría un modelo con requests en vuelo | Esperar o cambiar placement; no matar el proceso |
| Modelo a 0,27 t/s | Offload a CPU por `tensor_split` < 1 o swap | Revisar `CUDA_VISIBLE_DEVICES`, `--tensor-split`, `nvidia-smi` |
| `cache = -1` / `cached_tokens` ausente | No hay prefix cache real | Es correcto en EXL3/buun/vLLM sin cache; no inventar el dato |
| Sin `slot print_timing` en journal | Solo llama-server lo emite | En EXL3/buun/vLLM usar `timings` del body y `/api/metrics/activity` |
| Contexto pequeño en el cliente | Metadata cacheada en el cliente | `curl /v1/models \| jq`; refrescar el cliente |
| Modelo desaparece tras migrar | Migración `heimdall-gateway → llm-server` incompleta | Revisar `catalog.json` nuevo y backups `*.pre-llm-server-migration-*` |
| Config corrupta o clave borrada | JSON inválido o merge incorrecto | `config-migrate` es idempotente y no sobrescribe; revisar `#` comments |
| Dos modelos se desalojan en bucle | Ambos necesitan la misma GPU | Asignación por render, largest-first round-robin; ver `docs/runtime-placement-policy.md` |
| KV cache reprocesa ~30k en un turno de 3k | `preserve_thinking=false` + plantilla posicional Qwen3.5 | Ver `docs/kv-cache-reuse-hybrid-qwen35.md` (checkpoints, LCP) |
| `llama-swap` descarga todos los modelos | Se escribió `config.yaml` en runtime | Nunca escribir runtime config; publicar rutas alternativas |
| El editor muestra `heimdall-gateway` en docs/skills | Nomenclatura antigua | El nombre actual es `llm-server`; `heimdall-gateway` solo como alias deprecado en código de compatibilidad |

## Reglas de diagnóstico

- Primero el dato, luego la hipótesis. Un log o un `curl`, no una teoría.
- Ante discrepancia entre lo que dice el agente y lo que dice el sistema, gana el sistema.
- Si dos intentos fallan, para y replantea con evidencia nueva; no hagas cambios aleatorios.
