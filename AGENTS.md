# AGENTS.md — LLM Server (llamacpp-stack)

> Guía para agentes de codificación. **Este fichero se inyecta en cada sesión.** Es un router:
> el detalle vive en `.opencode/memory/`, `.opencode/skills/` y `docs/`.

## 0. Protocolo obligatorio (antes de actuar)

1. Lee **`.opencode/memory/invariants.md`** y **`.opencode/memory/preferences.md`**. No son
   opcionales: son restricciones declaradas por el usuario. Si el código contradice una regla,
   gana el código y debes reportar la contradicción.
2. Si vas a **depurar**, lee también `known-issues.md` (los síntomas ya vistos están ahí).
3. Si el trabajo toca **placement, réplicas, dedup, reasoning, logs o motores**, lee
   `decisions.md`.
4. Si existe una skill para el workflow (§6), **lee su `SKILL.md` completo antes de ejecutar**.
4b. Si el trabajo usa vocabulario de motores, rutas o réplicas, consulta
   `.opencode/memory/glossary.md`. **Cuatro motores distintos: `beellama` ≠ `buun` ≠
   `exllama` ≠ `vllm`.** Confundirlos produce diagnósticos falsos.
5. Antes de decir "esto ya se probó / está así desde siempre", consulta `events.md`.

Trabajo trivial (typo, renombrar variable): basta el paso 1.

## 1. Qué es este repo

Gateway de inferencia local compatible con OpenAI. Descarga y registra GGUF de Hugging Face,
genera `config.yaml` para `llama-swap` desde `catalog.json` + `conf.json`, y enruta a
`llama.cpp`, `buun` (EXL3 nativo) o `vLLM` (beta). Incluye réplicas conversation-affine,
auto-context, MTP/speculative, placement GPU, logs/diagnóstico y cleanup de huérfanos.

Comando público único: **`llm-server`** (`llamacpp_stack/llamacpp_api_install.py:main`).
No hay alias legacy público (`heimdall-gateway` es alias deprecado, no lo uses en docs nuevas).

```
cliente OpenAI → :11435 gateway → :11436 llama-swap (+guard) → llama-server / buun / vllm-server
                               ↘ conf.json + catalog.json → config.yaml (generado)
```

**Dos entry points, misma responsabilidad de política.** `:11435` aplica dedup, budget de
reasoning, réplicas y tool-repair, y escribe `api-requests.log`. `:11436` es el router
`llama-swap` (servicio `llm-server-router`); los clientes que entran por ahí (Hermes, web UI)
**no pasan por el gateway**: un fix solo en `:11435` no les llega.

## 2. No negociables (resumen de invariantes)

1. **Contexto máximo siempre** (263–264K por slot). No lo reduzcas para que quepa.
2. **No bajar cuantización** de modelo ni de KV cache para encajar en una GPU.
3. **No dividir contexto**; con `parallel=N`, `ctx_size = N × ctx_por_slot`.
4. **No desactivar MTP/draft** en comparativas.
5. **Nunca afirmar rendimiento sin medirlo en esta máquina.** Cero cifras de terceros.
6. **Comparaciones 1 vs 1**: mismo modelo, quant, prompt y `max_tokens`; una variable por carga.
7. **`uv` para todo.** Nunca `pip`, nunca `venv` puro.
8. **Merge, no sobrescritura** en `conf.json`/`catalog.json`; preserva `api_key`, `idle_ttl`,
   `models-dir`, certs y puertos.
9. **Arregla la causa, no el síntoma.** Sin parches rápidos ni revertir el plan.
10. **No escribir `config.yaml` en runtime**: llama-swap v255 descarga todos los modelos.
11. **Nada hardcodeado por modelo** en Python (ni nombres ni ids).
12. **Todo parche se documenta con su fuente** para poder reaplicarlo.

Lista completa con motivo y fecha: `.opencode/memory/invariants.md`.

## 3. `uv` es obligatorio

| Acción | Correcto |
|---|---|
| Tests | `uv run pytest -q` |
| CLI | `uv run llm-server info` |
| Instalar deps | `uv add <pkg>` / `uv sync` |
| Instalar proyecto editable | `uv pip install -e .` |
| Script suelto | `uv run python tools/foo.py` |

`uv.lock` manda. Si `uv` no está, instálalo antes de continuar. Los `pip` del `README.md` son
fallback documentado para humanos, no para el agente.

## 4. Mapa mínimo

```
llamacpp_stack/
  cli.py                # shim <400 líneas; re-exports de cli/
  cli/                  # constants, env, models, parser, replica, server_commands,
                        # gateway, crash_bundle, raw_log, daemon
  _cli_impl.py          # implementación legacy (~20k líneas) — zona caliente
  install.py            # fuente de verdad del instalador (prompts/flags/resolve_*)
  exllama_server.py     # backend EXL3 legacy (en retirada)
  bundle/               # install_llamacpp_stack.sh, defaults YAML
  migrations/           # migraciones de conf.json
configs/history/        # snapshots de autotuning (guardar aquí cada conf aceptada)
docs/                   # DFLASH-SPEC, PERF-BASELINE, KV-cache, placement, install
.opencode/memory/       # MEMORIA DEL PROYECTO (versionada)
.opencode/skills/       # SKILLS DEL PROYECTO (única ubicación escaneada)
tests/                  # pytest; markers real_model
tools/                  # bench/patch/verify
```

Ficheros de instalación real (**no editar `config.yaml` a mano**):

| Propósito | user mode | system mode |
|---|---|---|
| Settings | `~/.config/llm-server/conf.json` | `/etc/llm-server/conf.json` |
| Catálogo | `~/.local/state/llm-server/catalog.json` | `/var/lib/llm-server/catalog.json` |
| Runtime | `~/.local/state/llm-server/config.yaml` | `/var/lib/llm-server/config.yaml` |
| Env | `~/.config/llm-server/llm-server.env` | `/etc/llm-server/llm-server.env` |
| Request log | `…/api-requests.log.YYYY-MM-DD[.partN]` (symlink al día activo; 3 días) | `/var/lib/llm-server/…` |
| Raw ring | `…/api-raw-requests.log` (10 últimas, 1 MiB, fallback `/tmp`) | `/var/lib/llm-server/…` |

## 5. Flujos canónicos

**Tests**
```bash
uv run pytest -q
uv run pytest tests/test_managed_commands.py -q
uv run pytest -v -k "replica or config_migrate"
uv run python tools/verify_llama_cmds.py
```
`@pytest.mark.real_model` usa `stories260K.gguf` (~1,2 MiB) vía `tests/_real_test_model.py`.
`LLM_SERVER_TEST_MODEL_OFFLINE=1` desactiva la descarga. El bench live exige
`RUN_REAL_AUTO_PERF_E2E=1` y CUDA.

**Instalación (asistida: seguir `docs/LLM_INSTALL.md` Q1–Q8, siempre `--dry-run` primero)**
```bash
uv run llm-server install --mode user --backend auto --models-dir /var/llamacpp_models --idle-ttl 300 --dry-run
uv run llm-server install --mode system --backend auto --models-dir /var/llamacpp_models
```

**Cambiar configuración**
```bash
uv run llm-server config-migrate   # añade claves nuevas, nunca sobrescribe
uv run llm-server update           # regenera config.yaml
systemctl --user restart llm-server-manager llm-server-router
uv run llm-server info && curl -s localhost:11435/v1/models | jq
```

**Ajuste de rendimiento aceptado** → guardar snapshot en `configs/history/` y entrada en
`events.md`.

## 6. Skills del proyecto

Ubicación única: **`.opencode/skills/<nombre>/SKILL.md`** (es el directorio que escanean los
agentes; `skills/` en la raíz **no** se escanea). Antes de ejecutar el workflow, lee la skill
completa — no improvises el procedimiento.

| Skill | Úsala cuando |
|---|---|
| `llm-server-ops-triage` | 502/500/429/503, stream cortado, modelo que no carga, latencia rara |
| `llm-server-remote-deploy` | desplegar/verificar en `mpperez3@192.168.110.41` |
| `llm-server-perf-bench` | comparar configuraciones, medir tok/s, investigar regresión |
| `llm-server-model-tuning` | ajustar `ctx`/`batch`/`split`/quant/spec sin romper invariantes |
| `llm-server-autotune` | añadir un modelo nuevo y derivar su configuración inicial |
| `llm-server-installer-update` | install/update/migración user↔system, componentes ya instalados |
| `llm-server-vllm-backend` | vLLM: parches, reasoning parity, draft vocab, `timings` ausentes |
| `llm-server-mmproj-vision` | imágenes/visión, mmproj, modo lazy, "no procesa imágenes" |
| `llm-server-kv-cache-diagnosis` | un turno corto reprocesa miles de tokens de prompt |
| `llm-server-request-forensics` | "guarda esta request", captura #N, respuesta malformada, fixture |
| `llamabench-config-advisor` | pedir "config óptima" para un modelo concreto |

El mismo índice, en formato inyectable, está en `.opencode/memory/skills.index.md`.

## 7. Cómo mantener la memoria

Al terminar trabajo significativo, actualiza **solo** lo que seguirá siendo cierto:

| Conocimiento | Fichero |
|---|---|
| Regla nueva del usuario | `.opencode/memory/invariants.md` |
| Decisión con motivo | `.opencode/memory/decisions.md` |
| Fallo/trampa descubierta | `.opencode/memory/known-issues.md` |
| Qué pasó y con qué resultado | `.opencode/memory/events.md` |
| Procedimiento repetible | `.opencode/skills/<n>/SKILL.md` |
| Explicación larga de subsistema | `docs/` + puntero |

Reglas: una entrada por hecho, con fuente (fecha/commit/sesión/log), alcance explícito, y
actualiza `updated:` del frontmatter. Nunca guardes secretos. Una memoria con datos viejos es
peor que no tener memoria. Detalle en `.opencode/memory/INDEX.md`.
`tests/test_project_memory.py` valida la estructura: si falla, la memoria está corrupta.

## 8. Convenciones

- Python 3.12, `setuptools`. Claves: `snake_case` en JSON, `kebab-case` en CLI
  (`docs/arg-hyphen-conventions.md`).
- Idempotencia en `install.py` (`detect_existing_mode`, `choose_default_swap_port`,
  preservación de `models-dir`/certs/puertos).
- Preguntar antes de escribir en disco, usar `sudo`, abrir puertos o descargar modelos.
- Commits concisos y sin secretos; revisar `git status` / `git diff --cached`.
- **No modificar la instalación global de opencode** (`~/.config/opencode`) para resolver
  problemas de este repo: los recursos del proyecto van dentro del proyecto.

## 9. Troubleshooting rápido

| Síntoma | Primer paso |
|---|---|
| 502 al cargar | `bundle_ref` del JSON + `*_with_bundle` en el log rotado |
| 500 `failed to start command` | `PATH` de llama-swap sin `~/.local/bin` |
| 429 | `LLM_SERVER_MAX_CONCURRENT_PER_MODEL`, no es fallo del motor |
| 503 `model_loading` | Otro load en curso; no forzar |
| 0,27 t/s | Offload a CPU: revisar `tensor_split`/`CUDA_VISIBLE_DEVICES` |
| Sin `slot print_timing` | Normal en EXL3/buun/vLLM; usar `timings` y `Activity` |
| Contexto que no cuadra | `curl :11435/v1/models`; el cliente puede cachear |

Detalle y causas raíz: `.opencode/memory/known-issues.md`, `README.md#Troubleshooting`,
`docs/LLM_INSTALL.md#8`.

## 10. Referencias

- `README.md` — overview, API, configuración, troubleshooting.
- `docs/LLM_INSTALL.md` — checklist canónico de instalación para agentes.
- `docs/runtime-placement-policy.md` — placement, warmer, eviction.
- `docs/DFLASH-SPEC.md` — MTP vs DFlash2, presupuesto de VRAM, parches.
- `docs/kv-cache-reuse-hybrid-qwen35.md` — por qué se reprocesa el prompt.
- `docs/PERF-BASELINE-4090-QWEN38-27B.md` — baseline medido y protocolo de medición.
- `docs/VLLM-BETA.md`, `docs/arg-hyphen-conventions.md`, `docs/flags_llamacpp`.

*Última actualización: 2026-11-12. Si añades una dependencia, usa `uv add` y commitea `uv.lock`.*
