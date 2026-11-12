---
memory_file: decisions
version: 1
updated: 2026-11-12
load: always
scope: repo
---
# Decisiones técnicas vigentes

Registro tipo ADR, deliberadamente corto. **Estado `activa` = se implementa así hoy.**
Si vas a cambiar una de estas decisiones, primero lee el motivo y luego propón el cambio.

## Placement y réplicas

**D1 · Placement por rutas pre-renderizadas — activa (2026-10-03)**
El placement GPU se decide en el *render* de `config.yaml`, publicando una ruta por GPU
(`<model_id>__gpu0`, `__gpu1`, …). El gateway elige ruta en tiempo de request.
*Motivo:* escribir `config.yaml` en caliente hace que llama-swap v255 descargue **todos** los
modelos y destruya las KV caches. Ver `docs/runtime-placement-policy.md`.

**D2 · `reserved_base_gpus` no depende de la ocupación viva — activa (2026-10-03)**
La reserva de GPUs para modelos base es una función pura del catálogo, no del estado runtime.
*Motivo:* un render dependiente de ocupación produce configuraciones no reproducibles y
bloqueos fantasma entre GPUs libres.

**D3 · Política general de GPUs, no reglas por modelo — activa (2026-10-19)**
Llenar GPUs ociosas con réplicas calientes; desalojar réplicas antes que bases; nunca
interrumpir una base con requests en vuelo.
*Motivo:* las reglas ad-hoc por modelo rompieron las réplicas de EXL3 (regresión 2026-10-19).

**D4 · Réplicas `exclusive_gpus`, opt-in — activa**
`replicas.enabled=true`, `max=2`, `placement=exclusive_gpus`, `safety_vram_mib=2048`,
`prefer_base_over_replica=false`, `max_idle_s=1800`. Una réplica fría solo se crea si la base
ya está sirviendo (o si no hay réplica caliente que atienda).

**D5 · Afinidad de conversación con chokepoint único — activa (2026-10-01)**
Toda escritura de afinidad pasa por `bind()`; con `min_dwell_s`, `max_transfers` y
`evacuate_cooldown_s`. Prohibido escribir afinidad directamente (el WIP que lo hacía se
descartó).

## Gateway

**D6 · Dedup singleflight per-process, TTL 600 s — activa, off por defecto (2026-09-11)**
Solo `:11435`, solo `POST`; clave = método+path+principal+stream+sha256(body canónico con
allowlist). El seguidor espera al líder (`max_wait_ms=2000`) y recibe copia profunda.
*Motivo:* los clientes reintentan y se duplicaba inferencia.

**D7 · Tool-repair desactivado por defecto y gobernado por flag — activa (2026-09-30)**
`experimental.chat_tool_continue_repair.enabled=false`. Cuando se activa, no debe bufferizar
el stream completo ni tocar el `thinking`.
*Motivo:* una versión bufferizaba tokens y rompía thinking (latencia hasta 20 min).

**D8 · `reasoning_budget: half_context` con reserva de salida — activa (2026-10)**
`min(half_ctx, gen_lim − min(1024, gen_lim/8))`; `gen_lim = min(max_tokens, predict=16384)`.
vLLM replica la misma fórmula con `thinking_token_budget` y fuerza cierre con
`reasoning_end_str`. *Motivo:* paridad entre motores y evitar que el thinking consuma la
respuesta.

**D9 · Un solo spelling de razonamiento hacia el cliente — activa**
vLLM emite `delta.reasoning`; el gateway lo renombra a `reasoning_content` (stream y
no-stream) y separa el frame que mezcla content+reasoning.
*Motivo:* clientes que esperan un tipo por frame cortaban la respuesta al primer token.

**D10 · Dos entry points, misma política — activa**
`:11435` gateway (dedup, budget, réplicas, tool-repair, escribe `api-requests.log`) y
`:11436` `llama-swap` (+guard, servicio `llm-server-router`). Clientes como Hermes usan
`:11436` y no pasan por el gateway. *Motivo:* fixes aplicados solo en `:11435` no llegaban.

## Observabilidad

**D11 · Log de requests rotado por día y por tamaño, 3 días — activa (2026-09-11)**
`api-requests.log.YYYY-MM-DD[.partN]` con symlink `api-requests.log` al día activo; prune de
3 días; fallback `/tmp`. **Ojo:** el symlink puede apuntar al fichero antiguo tras un split;
el activo es el `.partN` con mtime más reciente.
Raw ring de las últimas 10 requests sin procesar en `api-raw-requests.log` (1 MiB, fallback
`/tmp`).

**D12 · `slot print_timing` solo existe en llama-server — activa**
EXL3 (exllama legacy y buun) y vLLM no lo emiten. Para ellos, la fuente de verdad es
`timings` del body y `/api/metrics/activity`. El gateway no sintetiza `cached` ni `drafted`.

## Motores y modelos

**D13 · EXL3 nativo vía buun; exllama legacy en retirada — activa (2026-09-14)**
`quant_method: exl3` + `engine: buun` para safetensors nativos. No parchear el upstream de
exllama3. Ver `.omo/plans/exllama-deprecation-exl3-fork.md` y `docs/DFLASH-SPEC.md`.

**D14 · MTP en vez de DFlash2 con ctx alto en 24 GiB — activa (2026-10-01)**
No es una regla universal: con `ctx 262144` el drafter DFlash2 hereda el ctx del target
(`ctx_size_draft: 0`) y se come ~5 GiB de KV → OOM en una 4090. Con ese presupuesto, MTP es la
opción viable. Si cambia el ctx o la VRAM, la comparación se rehace. Ver `docs/DFLASH-SPEC.md`.

**D15 · `mmproj` lazy opt-in — activa (2026-10-01)**
`mmproj.default_mode=always` por defecto; `lazy` publica todas las rutas sin `--mmproj` y lo
engancha en la instancia elegida, con TTL (`vision_sticky_ttl_s=3600`). Nunca reescribir una
ruta viva (reinicia el proceso y pierde la KV cache).

**D16 · Tests reales con modelo mínimo versionado — activa (2026-10-05)**
`@pytest.mark.real_model` usa `ggml-org/models::tinyllamas/stories260K.gguf` (~1,2 MiB) vía
`tests/_real_test_model.py`; plantilla de chat como fixture. Nada de depender de un 27B en CI.

## Configuración

**D17 · `config.yaml` es artefacto generado — activa**
Se regenera desde `conf.json` + `catalog.json` en cada `update`. `config-migrate` añade claves
nuevas con defaults y **nunca** sobrescribe valores existentes; segunda pasada idempotente.
