---
memory_file: invariants
version: 1
updated: 2026-11-12
load: always
scope: repo
---
# Invariantes del proyecto

Reglas duras declaradas por el usuario. **No se violan sin permiso explícito en el turno.**
Cada una incluye el motivo y la fecha en que se estableció. Si el código actual contradice
una regla, gana el código: repórtalo y propón corregir esta lista.

## Contexto y calidad

1. **Contexto máximo siempre.** El usuario ejecuta producción a máximo contexto (263–264K por
   slot). No lo reduzcas para "que quepa" ni como primera palanca de optimización.
   *Origen: 2026-10-25 — "yo siempre quiero usar el máximo".*
2. **No bajar cuantización** (ni del modelo ni de la KV cache) para encajar en una GPU.
   *Origen: 2026-06-18, repetido 2026-09/10.*
3. **No dividir el contexto** entre GPUs ni entre slots. `ctx_size` es por slot cuando
   `parallel > 1`; un modelo con `parallel=2` necesita `ctx_size = 2 × ctx_por_slot`.
   *Origen: 2026-09-04 — "¿por qué divides el contexto? quiero el contexto 262K".*
4. **No desactivar draft/MTP** para comparar rendimiento; forma parte de la configuración real.
   *Origen: 2026-06-18 — "No deshabilitites draft!".*
5. **No usar yarn rope scaling.** *Origen: 2026-06-18.*
6. **`tensor_split` nunca < 1** por GPU salvo justificación medida: por debajo se descarga a
   CPU y el modelo cae a ~0,27 t/s. *Origen: 2026-09-09.*
7. **Un id de modelo no puede contener espacios.** *Origen: 2026-10-25.*

## Configuración e instalación

8. **`conf.json` y `catalog.json` se actualizan por merge, nunca por sobrescritura.**
   Las claves existentes del usuario (`idle_ttl`, `api_key`, `api_port`,
   `llama_server_defaults`, certs, `models-dir`) se preservan.
   *Origen: 2026-06-15 — pérdida repetida de configuración al reinstalar.*
9. **Si `api_key` ya existe, no se regenera** en install/update. *Origen: 2026-06-15.*
10. **`idle_ttl` debe persistirse** en cada install/update. *Origen: 2026-06-15.*
11. **`config.json` tolera comentarios `#`** y, ante JSON corrupto, informa la línea y no
    borra la configuración. *Origen: 2026-06-15.*
12. **Claves no normativas (`speculative-defaults`, `mtp-defaults`, `gpu_capacity_mib`, etc.)
    nunca llegan al `config.yaml` de llama-swap.** *Origen: 2026-06-15.*
13. **Modo user no requiere `sudo`**; modo system sí, y debe pedirlo en el momento correcto,
    no al final. *Origen: 2026-09-07.*
14. **Componentes ya instalados se pueden saltar** (opción "skip") sin reinstalar todo.
    *Origen: 2026-09-07 — "¿pero qué haces compilando ahora de nuevo?".*
15. **Cambios en el instalador se despliegan también al host remoto** sin que el usuario
    tenga que pedirlo. *Origen: 2026-09-07.*

## Rendimiento y medición

16. **Nunca afirmar rendimiento sin medición real en esta máquina.** Prohibido citar cifras de
    llamabench, GitHub o estimaciones como si fueran propias.
    *Origen: 2026-06-18 — "quiero que me des los TPS reales".*
17. **Comparaciones 1 vs 1**: mismo modelo, mismo quant, mismo prompt, mismo `max_tokens`,
    mismo `thinking`, un cambio de variable por recarga, ≥2 repeticiones, mediana.
    *Origen: 2026-06-18 / 2026-10-25.*
18. **Benchmarks con prompts reales del proyecto**, no relleno sintético.
    *Origen: 2026-10-26.*
19. **No decidir con < 1024 tokens de salida** (ver `llm-server-bench`).
    *Origen: 2026-09-12 — conclusión invertida por medir 512 tokens.*
20. **Pruebas en modo mantenimiento** (solo requests locales) cuando haya clientes usando el
    servicio, y restaurar después. *Origen: 2026-10-25.*
21. **Nada de trabajo en background que compita con el path de respuesta**: el usuario mide
    latencia. *Origen: 2026-10-25.*

## Código

22. **Nunca hardcodear nombres de modelos** en Python. *Origen: 2026-09-09.*
23. **No parchear el código upstream de exllama/llama.cpp**; usar el mecanismo del proyecto.
    *Origen: 2026-09-04.*
24. **Arreglar la causa, no el síntoma.** Sin parches rápidos; sin revertir el plan.
    *Origen: 2026-10-01 — "no quiero parches rápidos, quiero arreglos robustos".*
25. **Desactivar una feature se hace por flag de configuración, no rompiendo código.**
    *Origen: 2026-09-30 — "te pedí desactivarla de la configuración, ahora has roto el código".*
26. **Antes de implementar una feature de rendimiento, demuéstrala con una prueba real**
    (GGUF y EXL3), y hazla configurable (on/off, tamaño).
    *Origen: 2026-10-25 — "no vamos a montar todo esto y que luego no funcione".*
27. **Runtime config es de solo lectura**: no escribir `config.yaml` en caliente; publicar
    variantes de ruta. *Origen: 2026-10 — llama-swap v255 descarga todos los modelos.*

## Operación

28. **Preparar todo antes de parar el servicio**; nunca detener sin plan de vuelta.
    *Origen: 2026-10-25 — "te dije preparar todo! Antes de detener el servicio!".*
29. **`uv` siempre**: `uv run`, `uv pip install`, `uv sync`. Nunca `pip`.
    *Origen: 2026-09-04 y AGENTS.md §2.*
30. **Documentar los parches aplicados y los pendientes, con su fuente (URL)**, de forma que se
    puedan volver a aplicar. *Origen: 2026-10-26.*
31. **Guardar la configuración validada en `configs/history/`** tras cada ajuste aceptado.
    *Origen: 2026-09-02 — repetido ≥6 veces.*
32. **Un modelo por GPU en modo réplicas**; no repartir un mismo modelo entre dos GPUs cuando
    se busca concurrencia. *Origen: 2026-09-04.*
33. **Respuestas concisas.** El usuario pide resúmenes cortos; el detalle va a docs.
    *Origen: 2026-10-02 — "resume, sé conciso, tus respuestas son muy largas".*
