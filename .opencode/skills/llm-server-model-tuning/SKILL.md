---
name: llm-server-model-tuning
description: Ajuste de parámetros de un modelo en LLM Server (ctx, parallel, tensor_split, cache_quant, batch/ubatch, checkpoints, MTP/spec) sin violar los invariantes del proyecto. Úsala SIEMPRE que el usuario pida "optimizar", "subir tok/s", "que quepa en una GPU", "cuánto contexto puedo poner", "mueve carga a la otra GPU", o cuando haya que decidir entre MTP y DFlash, entre kvarn4 y kvarn8, o cambiar el reparto entre GPUs. Incluye qué palancas son reales por motor y cuáles son ruido.
---

# Ajuste de modelos — LLM Server

## 0. Antes de tocar nada

1. Lee `.opencode/memory/invariants.md` (contexto máximo, no bajar quant, no dividir ctx, MTP on).
2. Estado actual real, no el que crees:

```bash
python3 - <<'PY'
import json,pathlib
p=pathlib.Path.home()/'.local/state/llm-server/catalog.json'
for m in json.load(open(p)):
    so=m.get('server_overrides') or {}
    print(m.get('model_id'), '| engine', so.get('engine'), '| ctx', so.get('ctx_size'),
          '| par', so.get('parallel'), '| cache', so.get('cache_type_k'),
          '| split', so.get('tensor_split'))
PY
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
uv run llm-server ps
```

3. Si hay clientes, `uv run llm-server maintenance on`.

## 1. Qué palanca es real según motor

| Motor | Palancas efectivas | Ruido / no-op |
|---|---|---|
| `llama.cpp` (`llama-server`) | `ctx-size`, `parallel`, `batch-size`, `ubatch-size`, `threads`, `cache-type-k/v`, `flash-attn`, `keep`, `cache-ram`, `tensor-split`, `n-gpu-layers`, spec | — |
| `buun` (EXL3 nativo) | `exl3` quant, `-gs grid_size`, `-cs cache_size`, `-cq cache_quant`, `-tp`, `--fit`, mmproj, spec MTP | Verifica en `--help` del binario instalado antes de asumir |
| `exllama` legacy | `-gs`, `-cs`, `-cq`, `-tp`, `-ccs` | **`batch/ubatch/threads/numa/flash/keep/cache-ram` se ignoran** (`parse_known_args`) — no pierdas tiempo ahí |
| `vLLM` | `max_model_len`, `kv-cache-dtype`, `max_num_seqs`, `gpu_memory_utilization`, `compilation_config`, spec | `batch/ubatch` no existen |

Regla: antes de "optimizar" un flag, confirma con `--help` del binario real que ese flag hace
algo. En este repo ya se han perdido horas tocando no-ops.

## 2. Orden de palancas (de menos a más riesgo)

1. **Reparto entre GPUs** (`tensor_split`): normaliza a ≥1 (`1.2,0.8`, `1.3,0.7`). Bajar de 1
   descarga a CPU → 0,27 t/s. Verifica VRAM por GPU tras cada cambio.
2. **`batch_size`/`ubatch_size`** (llama.cpp/buun): sube si sobra VRAM; 8192/2560 y 9216/2048
   son puntos ya probados. Subir demasiado puede *bajar* el uso de VRAM (efecto contraintuitivo
   ya observado): no lo uses como señal de que "falta memoria".
3. **`checkpoints`/`ctx_checkpoints`** (16 → 32): mejora reutilización de KV en conversaciones
   largas; no baja contexto.
4. **`cache_quant` / `cache_type_k/v`**: es cuantización de KV cache. **No se baja** (invariante).
   `kvarn8` a 262K ya provocó OOM (23,1 GB + 512 MB) y se revirtió a `kvarn4`.
5. **`parallel`**: multiplica el uso de KV. `parallel=2` exige `ctx_size = 2 × ctx_por_slot`.
   No lo subas para "aprovechar" la GPU si el contexto por slot se rompe.
6. **Spec (MTP vs DFlash)**: MTP es la opción viable en 24 GB; DFlash2 hace OOM.
   No desactives spec para "ver si va más rápido" (invariante 4).

## 3. Cómo decidir si un cambio es mejora

- Un cambio de variable por recarga. Todo lo demás constante.
- La medición se hace con `llm-server-perf-bench`, no a ojo.
- Antes/después con el mismo prompt, mismo `max_tokens`, `thinking` on, ≥2 repeticiones.
- Mira también VRAM por GPU y `tok/s` agregado con concurrencia, no solo C1.

## 4. Aplicar el cambio

```bash
# 1. editar catalog.json (merge; no reescribir el fichero entero)
# 2. regenerar
uv run llm-server config-migrate
uv run llm-server update
systemctl --user restart llm-server-manager llm-server-router
# 3. verificar argv real
tr '\0' '\n' < /proc/$(pgrep -f 'llama-server|llama-server-buun' | head -1)/cmdline | tr '\n' ' '
```

**Nunca** edites `config.yaml` a mano ni en caliente. Si el motor no recoge el cambio, es
porque la ruta no se re-renderizó, no porque haga falta tocar el YAML.

## 5. Guardar el resultado

Si el ajuste se acepta:

```bash
cp ~/.local/state/llm-server/config.yaml \
   configs/history/$(date +%Y-%m-%d_%H%M)_<modelo>_ctx<...>_split<...>.yaml
```

Y añade una línea a `.opencode/memory/events.md` con la medición y el motivo. Sin esto, el
siguiente agente repetirá el barrido desde cero.

## 6. Errores ya cometidos en este paso

- Reducir contexto como primera palanca → prohibido.
- Subir `parallel` sin multiplicar `ctx_size` → contexto por slot a la mitad.
- Mover carga a la GPU "que va más lenta" sin mirar VRAM real → OOM.
- Declarar "mejora" sin medir con la skill de bench.
- Dejar el modelo en mantenimiento tras las pruebas.
