---
name: llm-server-ops-triage
description: Diagnóstico de incidencias del gateway LLM Server (llm-server). Úsala SIEMPRE que aparezca un 502, 500, 429, 503, "stream disconnected before completion", una respuesta cortada, un modelo que no carga, latencia anómala o un cliente que dice que no ve un modelo. Contiene el orden exacto de comandos, qué log mirar (incluida la trampa del symlink roto) y la tabla síntoma→causa. No improvises el orden ni reinicies servicios antes de leer esto.
---

# Triage de incidencias — LLM Server

Regla número uno: **primero el dato, después la hipótesis**. En este proyecto un reinicio a
ciegas destruye KV caches y cuesta minutos de recarga.

## 1. Foto del sistema (siempre, en este orden)

```bash
uv run llm-server info
uv run llm-server ps
curl -s http://127.0.0.1:11435/v1/models | jq '.data[].id'
curl -s http://127.0.0.1:11436/v1/models | jq '.data[].id'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

`info` te dice el modo (user/system), los puertos y si los servicios responden. `ps` te dice
qué modelo está cargado y con qué `ctx`. Los dos `/v1/models` son distintos: el de `:11435` es
el gateway (resuelve capacidades); el de `:11436` es llama-swap.

## 2. Localizar el log correcto (trampa habitual)

```bash
ls -la ~/.local/state/llm-server/api-requests.log*     # user
ls -la /var/lib/llm-server/api-requests.log*           # system
```

- `api-requests.log` **es un symlink** al fichero del día, pero tras una rotación por tamaño
  puede seguir apuntando al antiguo. **El fichero activo es el `.partN` con `mtime` más
  reciente**, no el que dice el symlink.
- Retención: 3 días. Si el incidente es de anteayer, aún está; de la semana pasada, no.
- `api-raw-requests.log` guarda las **últimas 10 requests sin procesar** (1 MiB, fallback
  `/tmp`). Es lo que necesitas para reproducir un 502 o una respuesta malformada.

```bash
uv run llm-server requests --lines 200
uv run llm-server logs --lines 200 --journal
uv run llm-server logs --lines 400 --journal --since "15 minutes ago"
```

`logs --journal` añade journal + hint de `nvidia-smi` y el **crash bundle** (journal ~12k,
`nvidia-smi` ~2k). Si hay 502, busca `bundle_ref` en el JSON de respuesta y `*_with_bundle` en
el log.

## 3. Tabla síntoma → causa

| Señal | Causa más probable | Acción |
|---|---|---|
| `502` + `bundle_ref` | El proceso del modelo murió al cargar: OOM, `ctx` excesivo, draft incompatible, mmproj | Leer el bundle; comparar VRAM pedida vs libre; no reintentar en bucle |
| `500 failed to start command 'vllm-server …'` | El `PATH` de llama-swap no incluye `~/.local/bin` | `tr '\0' '\n' < /proc/$(pgrep -f llama-swap)/environ \| grep ^PATH=` |
| `429 overloaded` | `LLM_SERVER_MAX_CONCURRENT_PER_MODEL` alcanzado | No es fallo del motor. Bajar concurrencia o subir réplicas |
| `503 model_loading` | Otro modelo está cargando en un set de GPUs en conflicto | Esperar. El primer load gana y no se interrumpe |
| `503` con "víctima ocupada" | El load evictaría un modelo con requests en vuelo | Esperar o revisar placement; **no matar** procesos |
| Respuesta cortada al primer token | Frame que mezcla `content` + `reasoning` | Verificar que el gateway separa el frame (D9 en `decisions.md`) |
| Respuesta cortada a los ~90 s | Buffering del tool-repair o del guard | `grep -c tool_repair` en el log; repair debe emitir sin bufferizar |
| Modelo a 0,27 t/s | Offload a CPU: `tensor_split` < 1 o swap | Revisar `CUDA_VISIBLE_DEVICES` y `--tensor-split` del `ps` |
| Sin `slot print_timing` | Normal en EXL3/buun/vLLM | Usar `timings` del body y `/api/metrics/activity` |
| El cliente no ve un modelo | Metadata cacheada o el cliente entra por `:11436` | Comparar ambos `/v1/models`; refrescar el cliente |
| Contexto menor del esperado | Cliente cachea, o `parallel` reparte | `curl :11435/v1/models \| jq` y revisar `ctx_size`/`parallel` en `catalog.json` |

## 4. Aislar el ruido de clientes

Antes de medir o depurar latencia:

```bash
uv run llm-server maintenance on          # solo requests locales
# … pruebas …
uv run llm-server maintenance off
```

`maintenance --model <id>` redirige todo a un modelo. Restaura siempre al terminar.

## 5. Distinguir gateway de router

El gateway (`:11435`) aplica dedup, budget de reasoning, réplicas y tool-repair.
El router (`:11436`) es llama-swap; los clientes que entran por ahí **no pasan por el
gateway**. Si el síntoma solo aparece con un cliente concreto, comprueba por qué puerto entra
antes de tocar código: el bug puede estar solo en el guard.

## 6. Antes de reiniciar

- `systemctl --user restart` puede dejar el proceso anterior con el puerto tomado. Para un
  arranque limpio: parar units, comprobar `ss -tlnp | grep -E '11435|11436'` y
  `pgrep -af llama-server`, matar restos y arrancar.
- Un reinicio recarga el modelo: pierdes KV caches. Si el problema no es de configuración,
  probablemente no merece la pena.
- Nunca edites `config.yaml` para "arreglar" en caliente: llama-swap descarga **todos** los
  modelos.

## 7. Salida esperada del triage

Un informe de 5 líneas: síntoma, evidencia (línea de log o JSON), causa, acción aplicada,
verificación. Si no hay causa confirmada, dilo: no presentes una hipótesis como diagnóstico.
