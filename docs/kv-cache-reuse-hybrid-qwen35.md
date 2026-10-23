# KV cache reuse, checkpoints y divergencia de prompt (Qwen3.5 híbrido)

> Investigación de 2026-10-04 sobre por qué una petición que solo añade ~3k
> tokens al final reprocesa ~30k de prefill (`restored context checkpoint`
> en 94787 en lugar de reutilizar el LCP de 122607).
>
> **Resumen:** el proxy no corrompe el prompt. Hay dos causas que se componen:
> (1) la plantilla de Qwen3.5 con `preserve_thinking=false` trunca el LCP en
> cada turno nuevo; (2) el modelo es **híbrido** (linear-attention recurrente +
> full-attention), y en híbridos el KV reutilizable nunca supera el checkpoint
> exacto más nuevo ≤ LCP.

## 0. Entorno de referencia

- Backend real: **buun fork** `0.4.1-dev`, commit `a9b1b12`
  (`~/.local/opt/llm-server/buun/bin/llama-server-buun.real`).
  Fuente: `~/.local/opt/llm-server/buun/src` (mismo commit).
  No es llama.cpp mainline (`llama.cpp-source/` está en `b9010` y no contiene
  `cache_plan`, `reusable`, `edit/divergence`).
- Modelo: `qwen3.8-27b-EXL3` (`/var/llamacpp_models/Qwen3.8-27B-EXL3-3.5bpw`),
  `architectures: Qwen3_5ForConditionalGeneration`, `text_config.layer_types`
  = 3× `linear_attention` + 1× `full_attention` (`full_attention_interval: 4`).
- Flags relevantes del proceso (antes de la config recomendada): `--ctx-size
  262144 --ctx-checkpoints 16 --checkpoint-min-step 1024 --cache-ram 16384
  --keep 20000 --parallel 1 -ctk/-ctv turbo4 --swa-full --cache-prompt
  --chat-template-kwargs '{"preserve_thinking":false}'`.
- Config recomendada aplicada a la familia qwen (2026-10-04):
  `ctx-checkpoints 24`, `checkpoint-min-step 512`, `cache-ram 32768`,
  `preserve_thinking: true` (ver §5/§7).
  El journal confirma: `swa_full is not supported by this model, it will be
  disabled` → **no es SWA, es híbrido/recurrente**.

## 1. Síntoma

```
cache_plan_s: selected slot by LCP similarity, f_sim_best = 0.974 (>0.100), f_keep = 0.946
edit/divergence sample (cached/incoming/lcp/reusable/rewind/append/cache_prompt)
  = (129583/125846/122607/122607/6976/3239/1)
restored context checkpoint (pos_min = 94787, pos_max = 94787, n_tokens = 94788, n_past = 94788, size = 149.626 MiB)
CHECKPOINT_ATTN_ONLY_TRIM p0=94788
```

Solo ~3.2k tokens nuevos, pero se reprocesan ~27.8k + 3.2k ≈ 30k.
`progress = 0.79` es engañoso: es la posición absoluta
(`slot.prompt.n_tokens()/task.n_tokens()`), no el % reutilizado. `n_tokens` de
`print_timing` es lo procesado en la fase.

## 2. Causa A — `preserve_thinking=false` + plantilla posicional

`chat_template.jinja:116`:

```jinja
{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + content }}
{%- endif %}
```

El assistant del turno **actual** (después del último mensaje `user`) se
renderiza con `<think></think>`; cuando el cliente añade un nuevo `user`, ese
mismo assistant queda **antes** del último `user` y se renderiza **sin** think.
Es decir: **cada turno nuevo reevalúa retroactivamente el render del assistant
anterior** y el LCP se corta justo ahí, forzando el rewind de toda la cola del
turno previo (assistant + tool responses).

El proxy fija ese flag en `conf.json`:

```json
"llama_server_family_defaults": {
  "qwen": {
    "chat_template_kwargs": "{\"preserve_thinking\":false}"
  }
}
```

→ se hornea como `--chat-template-kwargs {"preserve_thinking":false}` en el
comando del modelo. El proxy **no** muta `messages` de forma no determinista
(`llamacpp_stack/_cli_impl.py:17965-17993`: `_normalize_openai_message`, merge de
system al frente, merge de asistentes finales; `:18031-18034` construye
`upstream_payload`).

### Medición reproducible

Dos peticiones consecutivas (R2 = R1 + un `user` final), mismo modelo, medido
con `/apply-template` + `/tokenize`:

| `preserve_thinking` | tokens A | tokens B | LCP | rewind | append |
|---|---|---|---|---|---|
| **`false`** (config actual) | 12070 | 12288 | **11215** | 855 | 1073 |
| `true` | 12090 | 12312 | **12086** | **4** | 226 |

Tokens divergentes decodificados (`/detokenize`):

```
A (R1): <|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>…
B (R2): <|im_start|>assistant\n<tool_call>\n<function=read_file…
```

## 3. Causa B — híbrido: reuse = último checkpoint ≤ LCP

- `LLM_ARCH_QWEN35 ∈ llm_arch_is_hybrid` (`src/llama-arch.cpp:1130`).
- Los checkpoints de híbrido son **estado puntual** (`pos_min == pos_max`),
  capturado en la frontera exacta: `server-context.cpp:18969-18985`.
- Selección: el checkpoint viable **más nuevo** con `pos_max < next_position`
  (`server-cache-plan-authority.cpp:550-575`); en híbrido no vale reanudar el
  estado recurrente (gated-deltanet) desde un token arbitrario.
- Trigger del restore: `pos_min >= pos_min_thold` (`server-context.cpp:17873-17875`).
- Al restaurar: `pos_next = min(pos_next, max(pos_min+1, pos_max))`,
  `n_past = min(…, n_tokens)` (`server-context.cpp:18252-18253, 18359`).
- Invalidación por turno: se borra todo checkpoint con
  `n_tokens > n_past_common` (el LCP real) (`server-context.cpp:18486-18505`).
- Creación solo en fronteras de turno `user` / cerca del final del prompt
  (`server-context.cpp:18880-18885, 18941-18943`). Con el template Qwen3.5, los
  `tool_response` se renderizan dentro de `<|im_start|>user`
  (`chat_template.jinja:147-158`), así que **los turnos de herramienta sí
  generan checkpoints**.

Efecto conjunto: el cliente reescribe el turno anterior (Causa A) → los
checkpoints de final de prompt quedan `> LCP` y se borran → solo sobreviven los
de frontera de turno ≤ LCP → si los turnos son largos, el hueco entre el último
checkpoint y el LCP es grande → reproceso grande.

En la prueba sintética R1 creó checkpoints en 48, 2050, 4052, 6054, 8056,
10058, 11041, 12065; R2 (LCP 11215) usó el de 11042 y reprocesó 1246. En el
incidente real el anillo se quedó clavado en 94787 durante ~40 min y ~34 tareas;
al publicar checkpoints nuevos (109330…125865) los restores pasaron a 125865 y
el prefill se redujo a cientos de tokens.

## 4. Herramientas

- Captura del payload completo del proxy:
  `experimental.tool_intent_capture = {enabled:true, mode:"all", path:…,
  max_entries, max_chars}` (`_cli_impl.py:1039-1110`). `mode:"all"` captura toda
  petición (no solo fallos de tool). Requiere reiniciar solo `llama-server-manager`.
- Comparador:
  `tools/compare_prompt_captures.py --dir <capturas> --base-url <llama-server> --tokenize`
  → diff de mensajes, prefijo común en caracteres y **LCP exacto en tokens**
  (vía `/apply-template` + `/tokenize`), que coincide con el `lcp` del journal.
- Log del prompt renderizado del motor: flag `--log-prompts-dir PATH`
  (`common/arg.cpp:5063`) vía `server_overrides`; requiere `update` + restart
  (descarga modelos).
- Journal: `journalctl --user -u llm-server-router` filtrando
  `edit/divergence sample`, `restored context checkpoint`,
  `created/erasing old context checkpoint`, `CHECKPOINT_ATTN_ONLY_TRIM`.

## 5. Palancas (requieren ventana: `config-migrate`/`update` + restart)

1. `llama_server_family_defaults.qwen.chat_template_kwargs`:
   `{"preserve_thinking": true}` (o omitirlo). Es la recomendación de facto para
   uso agéntico (ver §7). Trade-off real pero neto positivo para agentes:
   conserva el historial de razonamiento en el prompt (más tokens) y puede
   variar algo el comportamiento; a cambio el prefijo es estable, el LCP sube
   (medido: 11215 → 12086, rewind 855 → 4) y mejora la continuidad/calidad del
   tool-calling multi-paso.
2. Densidad de checkpoints: `ctx_checkpoints` 16 → 24–32 y
   `checkpoint_min_step` 1024 → 256, para cerrar el hueco LCP↔checkpoint. Es la
   config recomendada por la comunidad para agentes (ver §7); con 16 se sigue
   reproduciendo el problema.
3. `--keep 20000` y `--swa-full` **no** son la causa aquí.

## 6. Verificación end-to-end

1. `llm-server config-migrate && llm-server update` + restart del manager/router.
2. Reproducir dos turnos; comprobar en journal:
   - `edit/divergence` con `rewind` pequeño y `lcp` cercano a `incoming`;
   - `restored context checkpoint` con `pos_min` cercano al LCP;
   - `created context checkpoint` en fronteras de turno.
3. Comparar capturas reales:
   `python3 tools/compare_prompt_captures.py --dir …/prompt-captures --base-url … --tokenize`.

## 7. Qué dice la comunidad / upstream

Confirmaciones externas de las dos causas y de las palancas:

- **QwenLM/Qwen3.5 #48** — "Chat template breaks KV-cache reuse when
  `enable_thinking=false`". Mismo mecanismo; reportan ~90× de speedup y 99.4%
  de cache reuse al corregir la plantilla. Recomendación oficial: usar
  `preserve_thinking=true` junto con `enable_thinking=false` (el fix llega en
  Qwen3.6). <https://github.com/QwenLM/Qwen3.5/issues/48>
- **QwenLM/Qwen3.5 #112** — sobre cache en agentes. El maintainer confirma que
  Qwen3.5 son híbridos con linear-attention y que "el cache no se puede recortar
  como el normal"; llama.cpp/mlx usan checkpointing, y "preserving the reasoning
  content can reuse the prefix cache directly". Justo las dos causas A y B.
  <https://github.com/QwenLM/Qwen3.5/issues/112>
- **QwenLM/Qwen3.5 #131** — "if thinking is not preserved, then the last
  assistant turn will always break kv cache reuse". Además, los bloques
  `<think>` históricos vacíos generan prompt drift.
  <https://github.com/QwenLM/Qwen3.5/issues/131>
- **llama.cpp #22615** — contrapunto: `preserve_thinking=true` cambia la
  serialización (+43–193 tokens) pero **NO desbloquea el checkpoint cache en
  arch `qwen35`**; el skip híbrido `pos_min > pos_min_thold` sigue igual
  ("INERT", speedup 0.67×). Confirma que son **dos problemas independientes**:
  el flag arregla la Causa A (plantilla), no la Causa B (híbrido).
  <https://github.com/ggml-org/llama.cpp/issues/22615>
- **llama.cpp — checkpoints en fronteras de mensaje** (ya presente en buun:
  `message_spans`/`is_user_start`). Config recomendada para agentes:
  `--ctx-checkpoints 24 --checkpoint-min-step 256 --cache-ram 65536
  --chat-template-kwargs '{"preserve_thinking":true}'`. "With 8 checkpoints,
  he could still reproduce the issue. With 24, he worked for hours without
  problems." Nuestro 16/1024 está por debajo.
  <https://www.banandre.com/blog/how-a-single-llamacpp-pr-just-fixed-agentic-codings-worst-performance-bottleneck>
- **llama.cpp #25913** — en híbridos el estado recurrente no se puede rebobinar
  parcialmente: el checkpoint es el único rollback, así que el reuse queda
  acotado por `--ctx-checkpoints`/`--checkpoint-min-step`.
  <https://github.com/ggml-org/llama.cpp/issues/25913>
- **Featherless / unsloth** — para agentes multi-paso, "keep reasoning enabled
  and preserved across tool calls, or tool-calling and multi-step task quality
  degrade"; `preserve_thinking=true` está **recomendado para escenarios de
  agente**, no es solo un truco de cache.
  <https://featherless.ai/docs/chat-template-kwargs>

**Síntesis:** `preserve_thinking=true` es la configuración recomendada para uso
agéntico (calidad + cache) y **hacen falta las dos palancas**: el flag
(plantilla/ Causa A) **y** la densidad de checkpoints (híbrido/ Causa B).

### Más hallazgos de la comunidad (2026-10-04)

- **llama.cpp #24055** "Context checkpoints always invalidated on
  hybrid/recurrent models": confirma la Causa B y su historia de fixes:
  #24176 (checkpoints en cada mensaje `user`, no solo el último), #24411 (no
  borrar checkpoints más allá de `pos_next`), #25472 (caso acotado por ubatch),
  #25592 (registrar `pos_min = pos_max` para híbridos y restaurar solo en la
  posición exacta — **seguía open**), #26004 (persistir checkpoints en
  save/restore). El fork **buun** ya implementa el comportamiento de
  #25592 (`checkpoint_exact_frontier`, checkpoints `pos_min == pos_max`), que es
  justo el que explicamos en §3. <https://github.com/ggml-org/llama.cpp/issues/24055>
- **llama.cpp #22384** — el PR de fix del checkpoint híbrido (búsqueda por
  `pos_max <= pos_next`); ggerganov lo rechazó diciendo "usa
  `preserve_thinking`", y los testers demostraron que el flag **no** arregla el
  skip híbrido. Enlaza **spiritbuun/buun-llama-cpp#26** — el fork que corremos.
  <https://github.com/ggml-org/llama.cpp/issues/22384>
- **llama.cpp #22615** — ya citado: el flag no desbloquea el cache híbrido.
- **NousResearch/hermes-agent #56004** — el cliente descarta `reasoning_content`
  en el replay y el modelo "olvida" su razonamiento; los bloques `<think>`
  **vacíos** también perjudican el prefix-cache (cf. Qwen3.5 #131). El flag
  `preserve_thinking` es el control correcto server-side, pero el cliente debe
  reenviar el razonamiento. <https://github.com/NousResearch/hermes-agent/issues/56004>
- **QwenLM qwen-code #5637** — activa `preserve_thinking` por defecto en
  DashScope. Matriz medida: sin flag, turno 2 **BLIND** (0/3 recuerda); con
  flag, **PASS** (3/3). Coste: los `reasoning_content` históricos cuentan como
  input tokens. <https://github.com/QwenLM/qwen-code/pull/5637>
- **gpumod (jaigouk) — 2026-05-24** — `preserve_thinking` vs `enable_thinking`
  son equivalentes en single-shot; divergen solo en turnos con
  `reasoning_content` previo. Decisión de producción de Hermes-agent:
  `preserve_thinking: true`. <https://jaigouk.com/gpumod/research/20260524_preserve_thinking_findings/>
- **Towards AI (Qwen3.6-35B-A3B)** — coste cuantificado:
  `preserve_thinking` puede consumir 50–80K tokens de contexto en razonamiento
  a lo largo de ~15 turnos; para sesiones largas hay que gestionar el
  presupuesto de contexto. <https://pub.towardsai.net/your-ai-agent-is-goldfish-brained-qwen3-6-35b-a3b-is-the-fix-b6a687c2094a>
- **QwenCloud docs** — `preserve_thinking` soportado por la serie Qwen3.8;
  "en flujos multi-turno con tool calls, incluir el `reasoning_content` del
  assistant al devolver resultados de herramientas. Omitirlo degrada la
  precisión." <https://docs.qwencloud.com/developer-guides/text-generation/thinking>

## 8. Gateway: `reasoning_content` se descarta salvo que el modelo preserve thinking

`_normalize_openai_message` (`llamacpp_stack/_cli_impl.py:12276`) reconstruye cada
mensaje conservando **solo** `role`, `content`, `tool_calls`, `tool_call_id` y
`name`, y descartaba `reasoning_content`. Con `preserve_thinking=true` eso deja
la plantilla renderizando ` thinking` vacíos: estabiliza el LCP (medido
11215 → 12086) pero pierde el beneficio de continuidad de razonamiento
(qwen-code #5637: BLIND → PASS).

**Implementado y probado (2026-10-05):**

- `_normalize_openai_message(message, preserve_reasoning_content=False)`:
  reenvía `reasoning_content` (o `reasoning` como fallback) solo cuando el flag
  está activo.
- `_model_preserve_thinking_enabled(model_entry, args)` resuelve las opciones
  efectivas del modelo (defaults → familia → overrides) y comprueba
  `chat_template_kwargs.preserve_thinking`; solo para backend `llama.cpp`.
  Así el comportamiento depende de cómo esté configurado el modelo: qwen
  (preserve true) reenvía; glm/gemma (sin flag) no.
- Log diagnóstico `openai_chat_reasoning_content_forwarded`.
- Tests: `tests/test_normalize_reasoning_preserve.py` (13 casos) + suite
  relacionada → **131 passed**.
- E2E por el gateway: con `reasoning_content` que contenía un codeword oculto,
  el modelo lo recordó (`PLUMBUS-42`) y `/apply-template` mostró el contenido
  dentro de ` thinking`.

**Gotcha de despliegue:** el servicio (`llm-server-manager`/`-router`) importa el
**snapshot** `~/.local/opt/llm-server/python/llamacpp_stack` (ver
`docs/DFLASH-SPEC.md` §8), no el repo. Editar el repo no basta: hay que
re-sincronizar con `llm-server install --mode user` o parchear el snapshot y
borrar su `__pycache__` (lo que se hizo aquí). El snapshot quedó con este parche.

---

*Referencias de código del fork buun (commit a9b1b12):*
`tools/server/server-context.cpp`, `tools/server/server-cache-plan-authority.cpp`,
`src/llama-arch.cpp`; plantilla del modelo
`/var/llamacpp_models/Qwen3.8-27B-EXL3-3.5bpw/chat_template.jinja`;
proxy `llamacpp_stack/_cli_impl.py`.
