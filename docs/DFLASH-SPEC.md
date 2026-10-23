# DFlash / MTP — presupuesto de VRAM, trampas y compatibilidad

Guía canónica (medida, no teórica) para decidir entre **MTP** (`draft-mtp`) y **DFlash2** (`draft-dflash`)
como drafter especulativo en réplicas `buun`/EXL3, y para no comerse un OOM en GPUs de 24 GiB.

Toda cifra de este doc procede de mediciones reales en este host (2x RTX 4090, 24564 MiB cada una,
build `buun 0.4.1-dev (build 1, commit a9b1b12)`), no de extrapolaciones.

---

## 1. TL;DR

| Pregunta | Respuesta |
|---|---|
| ¿DFlash2 cabe en una 4090 de 24 GiB con el target EXL3 27B @ ctx 262144? | **Sí**: medido 20.199 MiB usados / 4,3 GiB libres **con mmproj off + `--no-logits-all` + `ctx_size_draft 2048` + draft Q3_K_M**. Mismo montaje con MTP: 19.533 MiB. |
| ¿Por qué se comes un OOM casi siempre? | `ctx_size_draft: 0` **hereda** el ctx del target (262144) → 5 GiB de KV del drafter. |
| ¿Vale la pena apagar visión para DFlash? | Sí: el mmproj EXL3 nativo son ~879 MiB de pesos + buffers ≈ 1.050–1.400 MiB de VRAM. |
| ¿MTP cuesta VRAM extra? | No *archivo* aparte ni `ctx_size_draft`, pero **sí** RAM: la cabeza `blk.64.nextn` (≈2,4 GiB bf16) + buffers del draft graph. Medido: **+2,2 GiB** sobre sin-draft. |
| ¿Cuál es más rápido? | **MTP en este montaje.** Medido en 1x4090 (§6.1): MTP 1,35x–2,03x, DFlash2 sólo 1,15x–1,18x. **Usa MTP.** Ojo: es un resultado *de esta config* (target EXL3 27B + drafter GGUF), no una ley — ver §6.2. |
| ¿Por qué DFlash2 no es más rápido aquí si en otros sitios lo es? | Su draft+verify cuesta **14,4–16,6 ms/token draftado** cuando le tocarían ~1,4 ms por tener 1,9B params contra 27B: **10x–12x sobre el coste teórico**. No es que el drafter sea grande, es que el camino de verificación no cobra su coste. Ver §6.3. |
| ¿Subir `spec_draft_n_max` en MTP acelera? | **No.** Medido a ctx 33.617: `n_max 3` = 90 tok/s, `n_max 5` = 83–86 tok/s, pese a que `n_max 5` acepta **más** (89% vs 82%). El `3` del catálogo es correcto. Ver §6.2. |
| ¿DFlash2 + EXL3 funciona? | Sí, **verificado en CUDA** (arranque + generación, sin asserts; 59–91% aceptación). El drafter es GGUF y el target EXL3, ambos en el mismo binario buun. **Pero revisa §6.5: con prefill largo devuelve HTTP 500.** |
| `--logits-all` cuesta | ~1.016 MiB de VRAM por réplica a batch/ubatch 1024. Ver §5. |
| Bonus con `--no-logits-all` + visión off | La réplica EXL3+MTP pasa de 21.548 → **19.533 MiB** usados: de 2,5 a **4,8 GiB libres**. |
| ¿DFlash2 aguanta ctx largo? | **NO con target EXL3**: HTTP 500 en el prefill de un prompt de 44.420 tokens (`llama_decode(ctx_dft) fused inject failed rc=1`). Causa: el drafter prefillia *chunked by ubatch* contra su KV, así que necesita `ctx_size_draft >= longitud del prompt` — y al subirlo, DFlash2 sigue perdiendo (52,9 vs 69,6 tok/s de MTP). Descargar tensores a CPU (`-otd`) no arregla nada. Ver §6.5. |
| ¿Se puede acelerar el prefill subiendo `batch-size`/`ubatch-size`? | **No.** Medido con mmproj ON: 1.555 tok/s a `-b 2048 -ub 1024` (producción) vs 1.553 a `-b 4096 -ub 2048`, y `-ub 4096` ** peta por VRAM** (falta 483 MiB). `--swa-full` on/off tampoco cambia nada. El prefill es compute-bound a ~1.500 tok/s. Ver §10. |

---

## 2. Componentes medidos

### 2.1 Target — `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`

| Grupo | Bytes | GiB |
|---|---|---|
| `model.language_model.*` | — | **13,23 GiB** (13.535 MiB) |
| `model.visual.*` (mmproj nativo, 333 tensores BF16) | 921.460.192 | **879 MiB** |
| **Total en disco** (2 shards) | — | **14,09 GiB** (15,0 GB) |

Config de texto: 64 capas, 24/4 cabezas, `head_dim` 256, **vocab 248.320**, `full_attention_interval 4`, ctx 262144.
El bloque extra `blk.64.nextn.*` es la **cabeza MTP** y queda *unused* cuando se usa DFlash2
(dead weight en disco, pero no en VRAM si DFlash2 no lo inicializa).

### 2.2 Drafter — `Anbeeld|zl-lab/Qwen3.8-27B-DFlash2-GGUF`

Metadatos GGUF (parseados a mano, **ojo: en estos ficheros las longitudes de string son `u64`, no `u32`**):

```
arch            = dflash          block_count = 5
embd            = 5120            ffn         = 17408
attention heads = 32 / 8 KV       head_dim    = 128
sliding_window  = 2048            causal      = false
block_size      = 8               vocab_size  = 248.320
base_model      = "Qwen3.8 27B"   size_label  = 1.9B
```

| Quant | Pesos | Nota |
|---|---|---|
| Q2_K | 672,6 MiB | **descartada**: su entrada de catálogo tiene `auto_ctx_failed: true` (`min-failed:exit-1`, trace en `/tmp/llamacpp-qwen3.8-27b-dflash2-q2_k-ctx8192-1u0a1xyf.log`) |
| **Q3_K_M** | **874,3 MiB** | ← la que hay que usar |
| Q4_K_M | 1.090,1 MiB | cabe, pero roba margen que necesitas para otra cosa |

KV del drafter: **20 KiB/token**.

| ctx del draft | KV |
|---|---|
| 2.048 (contexto de un turno de tool-calling) | 40 MiB |
| 262.144 (heredado del target) | **5 GiB → OOM** |

---

## 3. La trampa: `ctx_size_draft`

En el build buun, `-cd/--ctx-size-draft` con valor **0 significa "hereda la capacity del target"**.
Como el target corre a ctx 262144, el drafter se levanta allocating **5 GiB** de KV → OOM en una 4090
aunque el drafter pese menos de 1 GB.

Un drafter especulativo **nunca necesita 262k de contexto**: trabaja sobre los últimos cientos de tokens.
El server acota los buffers por su cuenta al arrancar:

```
draft ctx buffers bounded: n_ubatch 256, n_batch 256 (... KV spans target ctx 2048;
override cap with GGML_DFLASH_DRAFT_UBATCH)
```

→ **Siempre** fija `ctx_size_draft` explícito. `2048` es el valor correcto para uso interactivo/chat.

---

## 4. VRAM: presupuesto por réplica (4090, 24 GiB, 1 GPU por réplica)

Estado real con EXL3 + visión ON + MTP corriendo (antes de apagar nada): **21.548 MiB usados / 2.511 MiB libres**.
Margen real ≈ 2,3 GB. Lo que cuesta/decide:

| Concepto | Coste | Cómo se quita |
|---|---|---|
| Pesos target EXL3 | 13,23 GiB | — |
| mmproj (visión) | ~1.050–1.400 MiB | `mmproj_mode: lazy` u `off` en `server_overrides` |
| **Logits** (`--logits-all`, default ON) | **1.016 MiB** | `logits_all: false` → `--no-logits-all` |
| KV target @ 262144 + `turbo4` + `cache_quant 4` | resto del presupuesto hasta 21,5 GB | bajar `ctx_size` |
| Drafter DFlash2 Q3_K_M | 874 MiB | usar Q2_K (rota) |
| KV drafter | 40 MiB @ `-cd 2048` / 5 GiB @ `-cd 0` | **fijar `-cd`** |

Combinación que cabe — **cargada y medida** el 2026-10-01 en 1x4090:
`mmproj off` + `logits_all false` + `ctx_size_draft 2048` + draft Q3_K_M → **20.199 MiB usados / 4,3 GiB libres**.
(mismo montaje con MTP en vez de DFlash2: 19.533 MiB / 4,8 GiB libres; ver §6.1)

### 4.1 Cómo apagar visión en este gateway

`llamacpp_stack/cli/vision.py` → clave `mmproj_mode` dentro de `server_overrides`:

- `always` — siempre `--mmproj`.
- `lazy` — ruta base **sin** `--mmproj` + hermana `<id>__vision` con el projector (evict idle). Recomendado: conservas visión bajo demanda.
- `off` — nunca mmproj; el modelo queda `text-to-text`.

Gate: `_mmproj_enabled_for()` en `cli/server_commands.py` (~:810).

### 4.2 Cómo cablear DFlash2 en el catálogo

`cli/server_commands.py` (~:594-632) ya mapea las claves de override → flags llama.cpp:

| Override | Flag |
|---|---|
| `model_draft` | `--model-draft` |
| `spec_type` | `--spec-type` |
| `ctx_size_draft` | `--ctx-size-draft` |
| `n_gpu_layers_draft` | `--spec-draft-ngl` |
| `spec_draft_n_max` / `n_min` / `p_min` | `--spec-draft-n-max` / `-n-min` / `-p-min` |
| `spec_draft_device` | `--spec-draft-device` |

```jsonc
"server_overrides": {
  "spec_type": "draft-dflash",
  "model_draft": "/var/llamacpp_models/Anbeeld/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q3_K_M.gguf",
  "ctx_size_draft": 2048,          // ← NUNCA 0
  "n_gpu_layers_draft": 999,
  "spec_draft_n_max": 3,
  "spec_draft_n_min": 0,
  "spec_draft_p_min": 0.75,
  "mmproj_mode": "lazy",
  "logits_all": false
}
```

---

## 5. `--logits-all` / `--no-logits-all`: ~1 GB gratis sin coste de velocidad

`--logits-all` reserva la matriz de logits para **todos** los tokens del batch:

```
248.320 vocab × 1024 (ubatch) × 4 B = 1.016 MiB
```

En chat/completions sólo se necesitan los logits del **último** token del batch. Desactivarlo no cambia
el resultado generado ni la velocidad; libera ~1 GiB de VRAM por réplica — que en una 4090 con EXL3 27B
es exactamente el margen que decide si DFlash2 entra o peta.

`logits_all` es un override del gateway (expuesto el 2026-10-01, antes no existía):

```jsonc
"server_overrides": { "logits_all": false }
```

Emite `--no-logits-all`. Implementación: `cli/server_commands.py` (`normalize_server_overrides` set de
bools, `_bool_pair`, tupla de keys emitidas) + `_cli_impl.py` (copia legacy, 3 puntos).

> ⚠️ **El servicio corre un snapshot, no el repo.** Ver §8 antes de asumir que un cambio tuyo está activo.

> ⚠️ **Colisión con el drafting.** Al activar el drafting, el server avisa:
> `W decode: --no-logits-all is set but batch requested 13 outputs (> n_seq_max = 2); consider removing
> --no-logits-all for this workload`. No rompe nada (los drafts verificados se muestrean por separado),
> pero si aparece ese warning y crees que pierdes calidad de sampling, el culpable es esto. Con
> `parallel: 1` y el server así, es esperado.

---

## 6. MTP vs DFlash2

| | MTP (`draft-mtp`) | DFlash2 (`draft-dflash`) |
|---|---|---|
| VRAM extra (medido) | **+2,2 GiB** (cabeza `blk.64.nextn` ~2,4 GiB bf16 + draft graph) | **+2,9 GiB** (874 MiB de pesos + sus graphs/scratch + KV del drafter) |
| Fichero aparte | no | sí (GGUF 1.9B, 5 bloques) |
| `ctx_size_draft` | no aplica | **obligatorio** (§3) |
| Aceptación (medida) | 83–99%, long. media 2,5–3,0 | 59% (prosa) / 91% (repetitivo), long. media 2,25 / 8,24 |
| Coste por ciclo de draft | ~29 ms (depth 3) | ~38 ms (depth 2) / **~138 ms (depth 8)** ← superlineal |
| Throughput (medido) | **65–102 tok/s** | 58–60 tok/s |
| Requisito especial | ninguno | patch EXL3 side-tensors (§7) |

**Veredicto medido: usa MTP.** Es 1,7x más rápido que DFlash2 en la misma GPU, consume menos VRAM y
no necesita ni fichero extra ni `ctx_size_draft` ni apagar visión. DFlash2 sigue siendo útil como
*rejilla de seguridad* para drafting de una familia sin `nextn` en el target (ahí no hay alternativa),
pero en Qwen3.8-27B-EXL3 **pierde**: su overhead por ciclo se come la ganancia, y además **falla con
contexto largo** (§6.5). Ver §6.1 y §6.2.

⚠️ **Este veredicto es de *esta* configuración, no una ley.** El target es EXL3 y el drafter GGUF
(la clave aquí es el patch de side tensors, §7). Contra un target GGUF con su drafter nativo el
balance puede invertirse. No generalizar "MTP > DFlash2" fuera de este montaje.

Driver interno (log del server al arrancar con DFlash2):
`preselected adaptive shared DFlash2 driver (block_size=13, draft-max=<spec_draft_n_max>)` — el server lo
elige solo a partir de `size_label` 1.9B + `base_model` coincidente con el target. En el A/B de §6.1 la
telemetría del draft fue: `- n_max=8, n_min=0, p_min=0.00`, `- block_size=13, mask_token_id=248070, n_extract=5,
sample_from_anchor=true`, `- fused encoder+injection enabled`, `- device-staged capture->inject enabled`,
`- single-graph fused cycle enabled (inject rows=10, scratch seq=1)`. Esos últimos tres son los
sospechosos del coste superlineal por ciclo (§6.1).

---

## 6.1 Benchmark medido: MTP vs DFlash2 vs sin draft

**2026-10-01**, target `Qwen3.8-27B-EXL3-3.5bpw`, **1x RTX 4090** (`CUDA_VISIBLE_DEVICES=0`), ctx 262144,
`turbo4` + `cache_quant 4`, `--no-logits-all`, **sin mmproj**, 2 réplicas 512 tokens cada una,
`temperature 0.0` (greedy, determinista), `cache_prompt: false`. Script: `/tmp/opencode/bench_spec.py`
(no versionado; reproducirlo si quieres re-medir).

| config | tok/s (prosa) | tok/s (lista repetitiva) | aceptación | long. media | **VRAM pico** |
|---|---|---|---|---|---|
| sin draft | 48,4 / 49,9 | 50,4 / 50,1 | — | 1,00 | **17.313 MiB** |
| **MTP** | 65,3 / 66,5 | **102,2 / 101,7** | 83–99% | 2,5–3,0 | **19.533 MiB** (+2,2) |
| DFlash2 Q3_K_M | 58,9 / 60,2 | 58,7 / 58,1 | 59% / 91% | 2,25 / 8,24 | 20.199 MiB (+2,9) |

**Ganador: MTP.** 1,35x–2,03x sobre el baseline; DFlash2 sólo 1,15x–1,18x, con 0,7 GB más de VRAM.

### Por qué DFlash2 acepta más tokens y aun así es más lento

DFlash2 genera mucho más por ciclo (474–493 drafts frente a 319–340 de MTP) y su aceptación es
excelente (91% en la tarea repetitiva, `mean len = 8,24`), pero **el coste del ciclo DFlash2 crece de
forma superlineal con la profundidad del draft**:

| | ciclos para 512 tokens | ms/ciclo | ms/token | eficiencia |
|---|---|---|---|---|
| sin draft | 512 | 20,0 | 20,0 | 100% |
| MTP (`mean len` 2,97) | 174 | ~29 | 9,8 | **69%** |
| DFlash2 (`mean len` 8,24) | 63 | **~138** | 17,0 | 14% |
| DFlash2 (`mean len` 2,25) | 231 | ~38 | 16,9 | 53% |

El `verify/rollback histogram` del log lo confirma: los ciclos `8/8` (8 drafts, 0 rechazos, el mejor
caso posible) son **52 de 63 ciclos** y son precisamente los caros. Un ciclo de 8 tokens cuesta ~3,6x
uno de 2 tokens (138 ms vs 37,5 ms) → la ganancia en tokens/ciclo se la come el overhead. Sospecha: el
camino fusionado `encoder+inject` / `device-staged capture->inject` / `single-graph fused cycle` que
anuncia el log de arranque (`- fused encoder+injection enabled`, `- device-staged capture->inject
enabled`, `- single-graph fused cycle enabled`) re-inyecta y re-evalúa en lugar de reusar estado
incremental. Los costes por token draftado concretos están en §6.3.

⇒ **No bajes `spec_draft_n_max` esperando ganar**: bajar la profundidad de draft es la palanca
equivocada (el ciclo de DFlash2 ya es caro a cualquier profundidad: 37 ms @depth 2, 138 ms @depth 8).

## 6.2 `spec_draft_n_max`: subirlo en MTP **NO** es gratis

Medido con prompt de 33.617 tokens (`/tmp/opencode/bench_ctx.py`), mismo target, `temperature 0.0`:

| config | tok/s | speedup | aceptación |
|---|---|---|---|
| sin draft | 48,5 | 1,00x | — |
| **MTP `n_max 3`** | **90,0** | **1,86x** | 82% |
| MTP `n_max 5` | 82,8 – 86,0 | 1,71x–1,78x | **89%** |

MTP `n_max 5` tiene **más** aceptación (89% vs 82%) y aun así es **más lento**. La aceptación no es la
métrica que importa: lo que importa es cuántos tokens draftados se **verifican** por ciclo, y con
contexto largo cada verificación encarece el ciclo. Draftar más tokens de los que la GPU puede
verificar dentro del presupuesto del ciclo **pierde**. **Regla: `spec_draft_n_max 3` con ctx grande.**
El `3` del catálogo es el valor correcto; no lo toques por "más aceptación = más velocidad", porque esa
ecuación es falsa.

## 6.3 Por qué DFlash2 pierde aquí: 10x–12x su coste teórico por token draftado

El drafter DFlash2 tiene 1.9B params contra 27B del target → le tocaría ~1,4 ms/token draftado
(razón de pesos). Medido: **14,4–16,6 ms/token draftado**, o sea **10x–12x su coste teórico**. MTP
consume 4,5 ms/token draftado. La penalización no es "que el drafter sea grande", es que el camino
de draft+verify+inject no cobra su coste: `ms/ciclo` de DFlash2 es ~4,8x el de MTP (138 vs 28,7) con
sólo 2,8x más tokens por ciclo. Los flags de "fused path" que anuncia el log describen una
intención, no un graph medido. Con targets GGUF puros el balance puede ser otro; **esto es
específico del target EXL3 + drafter GGUF vía `ctx_other`.**

Ésta es también la explicación más probable de los benchmarks publicados de DFlash2 que dicen
"mucho más rápido": se miden con drafters GGUF contra targets GGUF, sin el patch de side tensors,
y a menudo con contextos cortos. Contra un target EXL3 27B con turbo4 y ctx largo, el overhead de
verificación domina y la ventaja se evapora.

⇒ **Conclusión accionable:** "MTP vs DFlash2" no es una pregunta con respuesta universal. En *este*
montaje (target EXL3 27B + turbo4 + ctx 262144 + drafter GGUF) gana MTP por 2x. Si algún día se
quiere DFlash2 en serio, la vía es un target GGUF con el drafter nativo de ese target, no este
combinado. Y no actives DFlash2 esperando el rendimiento que prometen los tweets: aquí está medido
a 1,18x.

## 6.4 Lo que se ve realmente en producción (y por qué "nunca llego a esas cifras")

Medido end-to-end por `POST :11435/v1/chat/completions` con API key, modelo `qwen3.8-27b-EXL3`,
2 réplicas tal cual las dejó el manager (`--spec-type draft-mtp --spec-draft-n-max 3 --ctx-size
262144 --predict 16384 --keep 20000 --no-logits-all`, una con `--mmproj` y otra con `--no-mmproj`).
Ojo con la métrica: el `tok/s` de abajo sale del log del servidor (`timings.predicted_per_second`,
que es **tiempo de decodificación puro**), NO de `completion_tokens / wall-clock` — ese último
incluye template, proxy y HTTP y da ~15-20% menos.

| | prompt | generados | **tok/s (servidor)** | tok/s (wall-clock) |
|---|---|---|---|---|
| `qwen3.8-27b-EXL3` | 15.937 | 175 | **76,9 / 81,6** | 64,8 |
| `qwen3.8-27b-uncensored-exl3-4bpw` | 15.937 | 207 | **79,4 / 82,1** | 61,6 |

Y con el **comando exacto de `config.yaml`** lanzado directamente (sin gateway), prompt de 33.617
tokens, `temperature 0.0` (`/tmp/opencode/bench_prodconf.py`):

| variante | tok/s |
|---|---|
| ctx 262144 (la de producción) | **83,3** |
| ctx 65536 | 84,4 |
| ctx 262144 sin `--keep` / `--cache-ram` | 83,9 |

Tres conclusiones:

1. **`--ctx-size 262144` NO penaliza.** 83,3 vs 84,4 tok/s: dentro del ruido. No bajes el ctx
   esperando velocidad; el ctx largo no es lo que te frena.
2. **MTP en producción va a ~77–83 tok/s**, no a 60. Los 60 eran un artefacto de medir mal
   (wall-clock en vez de tiempo de decodificación). Con eso, la cifra de §6.1 era el peor caso del
   A/B, no el tuyo.
3. **El primer request tras un evict paga la carga del modelo**: 6,4 tok/s medido end-to-end no es
   "MTP lento", es `llama-server` cargando 15 GB de pesos. Con `--idle-ttl` cortos + política de
   evict, el primer request tras un rato idle siempre es cold. Para medir velocidad de modelo,
   descarta el primer request.
4. **En salidas cortas el overhead domina**: peticiones de 16–35 tokens generados dan 35–51 tok/s
   aunque la decodificación va a 77+. Si tu uso real son respuestas cortas, la cifra que verás será
   ~40–50 tok/s y no es culpa del drafting.

⇒ **Respuesta corta a "¿mi conf actual me dará 90 tok/s?"**: no exactamente 90, pero **~77–83 tok/s
de decodificación real**, que es el 1,6x–1,7x sobre el baseline de 48,5. Los 90 del §6.2 eran otro
prompt. Los 60 que te dije antes estaban mal medidos.

### Bonus: el margen real que dejan las optimizaciones de §5

Con `--no-logits-all` + `mmproj_mode: off`, la réplica EXL3 27B + MTP baja de **21.548 MiB usados a
19.533 MiB** → de 2,5 GiB libres a **4,8 GiB libres**. Es lo que permite subir a 2 réplicas cómodas
o meter DFlash2 en algún escenario futuro sin pelear con el OOM.

---

## 6.5 DFlash2 + EXL3 **rompe** con prefill largo (HTTP 500)

Probado con el comando exacto de producción, **sin mmproj**, `temperature 0.0`, `n_predict 1024`, 2 runs,
`CUDA_VISIBLE_DEVICES=0`. Script: `/tmp/opencode/bench_big.py` (no versionado). El test grande
**paró `llm-server-manager` y `llm-server-router`** para dejar las GPUs libres.

| config | prompt corto (~200 tok), 1024 gen | prompt 44.420 tok, 1024 gen | acc | VRAM pico |
|---|---|---|---|---|
| sin draft | 52,0 tok/s | 47,2 tok/s | — | 16.933 / 17.127 MiB |
| **MTP `n_max 3`** | **68,2 tok/s** | **69,6 tok/s** | 81% / 84% | 19.493 / 19.709 MiB |
| DFlash2 `n_max 8` | 61,3 tok/s | **HTTP 500** | 58% | 20.157 MiB |

MTP gana también con queries grandes (68,2 vs 61,3 tok/s), y con menos VRAM. Pero lo importante es
la columna derecha: **DFlash2 no llega a generar.**

El fallo ocurre a los ~17,9 s, **durante el prefill** del prompt de 44.420 tokens — no durante el decode.
Con la línea de warning que lo precede (estaba oculta entre el ruido del log):

```
W decode: failed to find a memory slot for batch of size 512
E process: llama_decode(ctx_dft) fused inject failed rc=1 (n_tokens=512, offset=0)
  -> failed to process speculative batch
  -> decode() failed  ->  send_error 500
```

### Causa raíz: `n_ctx` del drafter, NO VRAM

En `llama-context.cpp:4620-4634`, `LLAMA_MEMORY_STATUS_FAILED_PREPARE` intenta `memory_update(true)`
una vez; si falla, loguea ese `failed to find a memory slot` y devuelve `rc=1`. Y el log de arranque ya
lo dice todo:

```
draft ctx buffers bounded: n_ubatch 512, n_batch 512
  (compute buffers scale with ubatch; drafter prefill chunks by ubatch; KV spans target ctx 2048;
   override cap with GGML_DFLASH_DRAFT_UBATCH)
```

El drafter **prefillia el prompt en trozos de `ubatch` (512) contra su propio KV de 2048 tokens**. Con
44.420 tokens se llena al cabo de ~2048 y `prepare` falla. No es VRAM: había 4,4 GiB libres.

⇒ La condición es `ctx_size_draft >= longitud del prompt`. Abajo la matriz que lo confirma.

### Descargar tensores a CPU no lo arregla (probado)

Flags que existen en el build buun: `--spec-draft-override-tensor`, `-otd`,
`--override-tensor-draft`, más `--spec-draft-type-k/-v-draft` y `--spec-draft-cpu-moe`.
La hipótesis era razonable (bajar presión de memoria) pero el 500 **no es de memoria**:

| config | `ctx_size_draft` | extra | resultado |
|---|---|---|---|
| cd2k | 2.048 | — | **HTTP 500** |
| cd2k_embcpu | 2.048 | `-otd 'token_embd\.weight=CPU'` | **HTTP 500** |
| cd8k | 8.192 | — | **HTTP 500** |
| **cd49k** | 49.152 | — | ✅ 52,95 tok/s (acc 56%) |
| **cd65k** | 65.536 | — | ✅ 52,79 tok/s (acc 56%) |
| cd65k_embcpu | 65.536 | `-otd 'token_embd\.weight=CPU'` | ✅ 52,71 tok/s (acc 56%) |
| **cd131k** | 131.072 | — | ✅ 52,67 tok/s (acc 56%) |
| cd131k_embcpu | 131.072 | `-otd 'token_embd\.weight=CPU'` | ✅ 52,42 tok/s (acc 56%) |

Script: `/tmp/opencode/bench_big2.py` (no versionado). El override no cambia nada: mismo 500 a 2k,
mismo resultado a 65k/131k. Ojo: llama.cpp **no loguea** qué tensor se descarga, así que no se puede
afirmar que `-otd` surtiera efecto — pero tampoco altera el resultado en ninguna dirección.

**Lo que sí lo arregla es subir `ctx_size_draft` a la longitud del prompt** — y eso es precisamente lo
que lo hace inviable: el KV del drafter son 20 KiB/token, luego 65k ≈ +1,25 GiB y 131k ≈ +2,5 GiB, y
con el prefill de un prompt de 44k te come el margen que libertaste apagando mmproj.

### Y aun arreglado, DFlash2 pierde

Con DFlash2 **funcionando** (cd49k/cd65k/cd131k) y el mismo prompt de 44.420 tokens:

| config | prompt 44.420 tok, 1024 gen | VRAM |
|---|---|---|
| **MTP `n_max 3`** | **69,6 tok/s** | 19.709 MiB |
| DFlash2 `n_max 8` | 52,4 – 53,0 tok/s | 20.157 MiB + KV del drafter |

⇒ **Veredicto práctico: usa MTP.** DFlash2 sobre EXL3 solo funciona si `ctx_size_draft` cubre el
prompt completo, y cuando lo cubre es ~1,32x más lento y consume más VRAM. El ctx largo es justo donde
`ctx_size 262144` aporta, así que son incompatibles. Si algún día se necesita DFlash2 de verdad, el
camino es un target GGUF con el drafter nativo de ese target, no EXL3 + drafter GGUF vía `ctx_other`.

---

## 7. Requisito de build: patch `buun-exl3-shared-head-side-tensors.patch`

DFlash2 **no crea su propia cabeza**: hace *borrow* del `lm_head` del target vía `cparams.ctx_other`.
Los formatos con side tensors estáticos (EXL3 `svh`/`suh`, NVFP4 `scale2`) los llevan como fuentes
`MUL_MAT` extra (**`output_s` → `src[2]`, `output_in_s` → `src[3]`**) en lugar de dentro del peso.
Sin propagarlos, el backend **asserts**.

`llamacpp_stack/bundle/patches/buun-exl3-shared-head-side-tensors.patch` parchea:

- `src/llama-model.cpp` (~:3743) — propaga `output_s` / `output_in_s` cuando `share_tensors` presta la cabeza.
- `src/models/dflash.cpp` — lm_head tomado del target vía `ctx_other`, scatter de draft-vocab reducido
  (mapa `d2t`), `build_lora_mm(output, cur, output_s, output_in_s)`.

El binario lo marca con `~/.local/opt/llm-server/buun/.bundled-patches.stamp`.
**Sin este patch, DFlash2 + EXL3 aborta en load.**

### Verificación realizada (smoke test)

CPU only, puerto 18099, `-ngl 0 -c 2048 -cd 2048 --no-mmproj --cache-ram 0`, target EXL3 + draft Q3_K_M:
arrancó sin asserts y generó con `draft_n:8, draft_n_accepted:8`. Log: `/tmp/opencode/dflash_smoke.log`.

### Verificación en GPU (2026-10-01)

El A/B completo de §6.1 se ejecutó **en CUDA** con el draft Q3_K_M → el path de side tensors EXL3
(`build_lora_mm`) **sí** se ejercita, arranca sin asserts y genera. Log del draft: `/tmp/opencode/bench_dflash2_q3.log`
(en `~/.local/state` no; está en `/tmp/opencode/`, no versionado).

⚠️ **El patch no es suficiente.** Con prefill largo (>44k tokens) el path DFlash2 revienta igualmente
con `llama_decode(ctx_dft) fused inject failed rc=1` — ver §6.5. El patch arregla el *load*; el 500 del
prefill largo es otro bug: el drafter prefillia *chunked by ubatch* contra su propio KV y se queda sin
`memory slot` cuando `ctx_size_draft` < longitud del prompt. Solo se esquiva subiendo `ctx_size_draft`
a ≥ el prompt, lo que lo hace inviable (más VRAM y más lento que MTP).

---

## 8. ⚠️ Nota de despliegue: el servicio NO ejecuta este repo

`llm-server-manager.service` → `~/.local/bin/llm-server-manager-start` → `exec $PYTHON_BIN -m llamacpp_stack.cli`
con (de `~/.config/llm-server/llm-server.env`):

```
PYTHON_BIN=/home/martin/.local/opt/llm-server/venv/bin/python
LLM_SERVER_PYTHONPATH=/home/martin/.local/opt/llm-server/python
```

Es decir, el daemon importa `~/.local/opt/llm-server/python/llamacpp_stack/`, un **snapshot copiado**
por `install.py:ensure_runtime_python` (`shutil.copytree`, ignorando `__pycache__`), **no** el árbol
de trabajo del repo.

Consecuencias prácticas:

- El `llm-server` del PATH (`.../OpenCodeAutoModelDiscover/.venv/bin/llm-server`, editable → repo) ve tu
  código nuevo; el servicio ve el snapshot viejo. **`llm-server update` re-renderiza `config.yaml` con el
  snapshot**, así que un flag nuevo del repo se pierde en silencio.
- El snapshot va además retrasado respecto al repo (le faltan `cli/vision.py` y `cli/replica_policy.py`,
  y difieren `_cli_impl.py`, `cli/server_commands.py`, `cli/replica.py`, `cli/__init__.py`, `cli/models.py`,
  `buun_install.py`).
- Workaround para un flag puntual (lo que se hizo con `logits_all` el 2026-10-01): parchear **solo** los
  puntos necesarios en el snapshot y borrar sus `__pycache__`; o re-ejecutar `llm-server install`
  (ojo: eso copia *todo* el paquete, incluido WIP sin commitear).
- Bugs latentes del snapshot, por si los ves: `_cli_impl.build_llama_server_command` no hace `return` en los
  branches y tiene un "leftover pass" que emite overrides desconocidos como `--key value` (por eso
  `logits_all` salía como `--logits-all false`, que llama.cpp ignora); `swa_full` se emite como
  `--swa-full <bool>` en vez de `--swa-full/--no-swa-full`.

Verificación estándar tras cualquier cambio de flag:

```bash
grep -c -- '--no-logits-all' ~/.local/state/llm-server/config.yaml   # esperado 2 (model + réplicas)
python - <<'PY'
import yaml, pathlib
c = yaml.safe_load(pathlib.Path.home().joinpath(".local/state/llm-server/config.yaml").read_text())
print([k for k, v in c["models"].items() if "NO-LOGITS-ALL" in str(v).upper()])
PY
```

---

## 9. Runbook rápido

```bash
# ¿Cuánta VRAM libre hay ahora?
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

# ¿Está el patch DFlash aplicado?
grep -n "output_in_s" ~/.local/opt/llm-server/buun/src/src/llama-model.cpp | head
test -f ~/.local/opt/llm-server/buun/src/src/models/dflash.cpp && echo "dflash.cpp OK"
cat ~/.local/opt/llm-server/buun/.bundled-patches.stamp

# ¿Qué draft devices eligió el server? (log del server, no el journal del manager)
grep -E "preselected adaptive|draft ctx buffers|drafter warmup" <server-log>

# ¿DFlash2 ha reventado en un prefill largo? (§6.5)
# el aviso REAL está una línea por encima del error:
grep -E "failed to find a memory slot|fused inject failed|failed to process speculative batch" <server-log>

# ¿cuánta capacity de KV tiene el drafter? (si es menor que el prompt, dará 500)
grep -o "KV spans target ctx [0-9]*" <server-log>
```

---

## 10. Barrido de prefill: `batch-size` / `ubatch-size` / `swa-full` no mueven la aguja

Medido con la entry **`__vision`** (mmproj ACTIVADO, el peor caso de VRAM) + MTP `n_max 3`,
`temperature 0.0`, `cache_prompt=false` (prefill en frío), prompt de **44.420 tokens**, `n_predict 128`,
`CUDA_VISIBLE_DEVICES=0`. Scripts: `/tmp/opencode/bench_prefill.py` y `bench_prefill2.py` (no versionados).
Servicios `llm-server-manager`/`llm-server-router` parados durante el test.

| `-b` | `-ub` | `swa-full` | prefill tok/s | prompt_ms | decode tok/s | VRAM pico |
|---|---|---|---|---|---|---|
| **2048** | **1024** | **ON** | **1.555** | 28.557 | 84,0 | **21.073 MiB** ← producción |
| 2048 | 1024 | ON (rerun) | 1.511 | 29.392 | 77,2 | 21.073 MiB |
| 4096 | 2048 | ON | 1.553 | 28.607 | 77,0 | 22.547 MiB |
| 4096 | 4096 | ON | **HTTP 500** | — | — | OOM real |
| 8192 | 4096 | ON | **HTTP 500** | — | — | OOM real |
| 2048 | 1024 | off | 1.514 | 29.346 | 77,6 | 21.073 MiB |
| 2048 | 2048 | off | 1.506 | 29.506 | 73,2 | 22.547 MiB |

**Conclusión: la config de producción ya es la óptima.** Doblar `ubatch` da −0,2% de prefill (ruido;
la varianza entre corridas es ±3%), cuesta +1.474 MiB de VRAM y cuadruplicarlo ** peta por VRAM de verdad**:

```
E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 483.63 MiB on device 0: cudaMalloc failed: out of memory
E ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 507126016
E ggml_backend_cuda_graph_compute: CUDA pool allocation failed (out of VRAM), failing graph compute
E llama_decode: failed to decode, ret = -2
```

Con el pico de 24.059 MiB de 24.564 no queda ni medio GiB. Y `--swa-full` (que `conf.json` fuerza a
`true` sobre un default del build que es `false`) tampoco cambia nada: 1.514 vs 1.555 tok/s y **el mismo
VRAM**.

⇒ **El prefill de este modelo es compute-bound a ~1.500 tok/s en una 4090**, no está limitado por el
tamaño de lote ni por el tipo de caché de contexto. No hay palanca que ganar aquí; el cuello es la
matemática de 27B en fp4.

### Palancas de VRAM que quedan sin probar

- `--mmproj-gpu-swap`: "temporarily swap a reloadable speculative context (MTP or external)". Evicta la
  cabeza MTP de la VRAM en reposo y la recarga bajo demanda. A cambio: latencia en el primer token
  tras un rato de inactividad.
- Bajar `--ctx-checkpoints` (4) o `--cache-ram` (16384): RAM de host, no VRAM. No acelera nada.