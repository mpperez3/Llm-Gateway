---
memory_file: events
version: 1
updated: 2026-11-12
load: on_demand
scope: repo
---
# Memoria de eventos

Cronología de lo que pasó, lo que se decidió y lo que se midió. **Más reciente arriba.**
Formato: `fecha · evento · resultado · evidencia`.

Este fichero es un log, no un documento: se añade, no se reescribe. Mantener ≤ 60 entradas;
al pasar de ahí, mover las más antiguas a `events-<año>.md`.

## 2026-11

- **2026-11-12** · Se crea este sistema de memoria y las skills del proyecto a partir del
  análisis de 45 conversaciones (353 sesiones, 27 279 comandos). · Resultado: AGENTS.md pasa a
  router, memoria en `.opencode/memory/`, skills consolidadas en `.opencode/skills/`.
- **2026-11-10** · Comparativa NInfer vs AutoRound y revisión de config PCI tras reboot. ·
  Resultado: routing sticky por réplica; NInfer sirve rutas calientes tras arrancar
  llama-swap. · Evidencia: commits `b8daa09`, `5f2c7b6`, `2bf3634`.
- **2026-11-02** · Respuesta cortada en cliente; captura `#57` del guard. · Resultado: se
  documenta el corte por frame mixto vLLM y se separa en dos deltas.

## 2026-10

- **2026-10-31** · Una respuesta salió en chino (síntoma de calidad). · Resultado: se guarda el
  prompt completo como fixture de regresión antes de la respuesta.

- **2026-10-28** · Se aplica el parche 5 de vLLM + canario de stream. · Resultado: decisiones
  sobre speculative decoding y `--kv-cache-dtype auto` documentadas.
- **2026-10-26** · Tuning de kernel/receta y despliegue vLLM; paridad de `max_tokens` y
  reasoning con buun. · Resultado: `docs/VLLM-BETA.md` + parches con fuente.
- **2026-10-25** · Cambios de kernel: ¿mejoran la velocidad? · Resultado: se fija "máximo
  contexto siempre (264K)" y el protocolo de medición con prompts reales.
- **2026-10-20** · Se cargó una réplica con el modelo base activo habiendo un modelo en
  desuso. · Resultado: `prefer_base_over_replica` y política de eviction revisadas.
- **2026-10-19** · Regresión: EXL3 se quedó sin réplicas tras el cambio de política. ·
  Resultado: política general de GPUs, sin reglas por modelo. · Evidencia: `.omo/notepads/`.
- **2026-10-09** · Hermes solo detectaba un modelo. · Resultado: era configuración, no código;
  el contexto se reporta crudo (262144), sin reducir.
- **2026-10-05** · Tests con modelo real pasan a `stories260K` versionado. · Resultado: suite
  sin dependencia de un 27B.
- **2026-10-04** · Investigación de reutilización de KV cache en Qwen3.5 híbrido. · Resultado:
  `docs/kv-cache-reuse-hybrid-qwen35.md` (checkpoints, LCP, `preserve_thinking`).
- **2026-10-03** · El tool-repair rompía thinking y respuestas; se diseñó el placement runtime.
  · Resultado: repair por flag, streaming sin buffer; `docs/runtime-placement-policy.md`.
- **2026-10-01** · Se implementa mmproj lazy y se compara DFlash2 vs MTP. · Resultado: MTP
  viable; DFlash2 OOM en 24 GB. · Evidencia: `docs/DFLASH-SPEC.md`.
## 2026-09

- **2026-09-30** · Compatibilidad de `qwen3.8-27b-EXL3` con `mmproj`. · Resultado: se parchea
  buun para cargar mmproj nativo y se arranca la línea DFlash.

- **2026-09-29** · Limpieza de caché de Hugging Face y backups de migración. · Resultado:
  modelos ya presentes en `models/` no se re-descargan.
- **2026-09-16** · Migración `heimdall-gateway → llm-server` muy defectuosa (modelos, certs,
  token, binarios no portados). · Resultado: lista de errores convertida en invariantes 8–15 y
  en `known-issues.md`. **No debe repetirse.**
- **2026-09-15** · Renombrado a `llm-server` con alias deprecado. · Resultado: migración
  copy-if-missing + units nuevas.
- **2026-09-14** · Retirada de exllama legacy en favor de buun; barrido VBR/TCQ a 262K. ·
  Resultado: `cache_quant 4` y `grid_size` como palancas reales; el resto era ruido.
- **2026-09-13** · Respuestas malformadas de exllama (`useruser`, líneas en blanco). ·
  Resultado: fixtures de regresión + separación content/tool; kvarn8 revertido a kvarn4 por OOM.
- **2026-09-11** · Dedup de requests idénticas, rotación de logs y refactor del CLI monolito. ·
  Resultado: singleflight TTL 600 s, log diario por tamaño, `cli/` modular. · Evidencia:
  `.omo/plans/dedup-inflight-cache.md`, `.omo/plans/log-rotation-crash-diag.md`.
- **2026-09-09** · Con `parallel 2` no se procesaban dos queries simultáneas. · Resultado:
  causa en `ctx_size`/límite de concurrencia; probado con 10 peticiones concurrentes.
- **2026-09-04** · Cambio de `qwen3.8-27b-q4_1` a EXL3 3.5bpw; fallos de requests tras las
  23:00 por jinja/afinidad. · Resultado: backend exllamav3, jinja y logs corregidos.
- **2026-09-01** · beellama respondía 200 directo pero 500 vía gateway. · Resultado: ruteo y
  `PATH` del wrapper; concurrencia verificada.

## 2026-08 y anteriores

- **2026-08-31** · Se añade comando de update para dependencias registradas (llama.cpp,
  llama-swap, vLLM) y guía de instalación para LLMs. · Evidencia: `docs/LLM_INSTALL.md`.
- **2026-06-18** · `stream disconnected before completion` en el cliente. · Resultado:
  flattening de tools solo existía en el superserver; clientes en `:11434` no lo recibían.
- **2026-06-15** · Pérdida de configuración al reinstalar (`llama_server_defaults`, `idle_ttl`,
  `api_key`). · Resultado: merge no destructivo, `#` en JSON, error con línea. **Origen de los
  invariantes 8–12.**

## Protocolo

Añadir una entrada cuando: se despliega un cambio, se mide algo que contradice una creencia, se
corrige un incidente, o el usuario expresa una restricción nueva. Una línea, con evidencia
(commit, doc, log, sesión). No borrar entradas: mover a `events-<año>.md`.
