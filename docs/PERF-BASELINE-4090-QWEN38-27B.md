# Baseline de rendimiento — Qwen3.8-27B / RTX 4090

> Conocimiento acumulado de las campañas de medición de octubre-2026. **Léelo antes de
> tocar flags de rendimiento, KV cache o motores.** Todo lo que aparece aquí está medido
> en esta máquina, salvo lo marcado explícitamente como referencia externa.

Hardware: 2× RTX 4090 24 GB (driver 615.71.09, kernel 7.0.0-38, `gpu_unlock.targets=all`),
Ryzen 9 7950X3D, 192 GB RAM. Modelo: `Qwen3.8-27B-EXL3-3.5bpw` (safetensors EXL3, híbrido
Qwen3.5: 64 capas, 16 de atención completa + 48 Gated DeltaNet, `head_dim` 256, 4 KV heads).
Motor: `buun` (`spiritbuun/buun-llama-cpp @ 8172dd1`), `--parallel 1`, MTP `draft-mtp`.

---

## 1. Baseline de producción (ctx 262144)

Medido con el comando exacto de `config.yaml` en GPU libre, prompt sintético en frío
(nonce único → sin prefix cache), `temperature 0`, gen 128.

| escenario | prefill tok/s | decode tok/s | VRAM |
|---|---|---|---|
| ctx 262144, prompt ~199k, turbo4 | **1272** | **57–58** | 19.8 GB |
| ctx 262144, prompt ~32k, turbo4 | 1806–1810 | 84–88 | 18.6 GB |
| ctx 98304, prompt ~64k, turbo4 | 1675 | 75–78 | 15.6 GB |
| ctx 98304, prompt ~32k, turbo4 | 1804 | 78–84 | 15.6 GB |

Lectura: el decode cae ~32 % de 32k a 200k de contexto; el prefill cae ~30 %. El techo
teórico de decode a batch 1 es ~85 tok/s (15 GB de pesos por token ÷ ~1 TB/s), así que a
contexto corto **ya estamos en el techo** y solo el contexto profundo deja margen.

---

## 2. Cambio de driver/kernel (595.99.02 → 615.71.09) — medido

El 25-oct se instaló la serie 615 + `gpu_unlock.targets=all`. Efecto medido con los
`slot print_timing` del propio motor, comparando **solo tareas en frío** (`cached==0`) y a
**longitud de contexto idéntica**:

| contexto | antes (mismo día) | después | Δ |
|---|---|---|---|
| 8k | 1594 | 1748 | +9.7 % |
| 16k | 1671 | 1811 | +8.4 % |
| 24k | 1647 | 1786 | +8.4 % |
| 49k | 1548 | 1728 | +11.6 % |
| 98k | 1429 | 1562 | +9.3 % |

Serie horaria de la misma métrica (20k–100k, frío): `12h → 1529`, `13h → 1601`,
`15h → 1695`. El salto aparece ya en el arranque de 12:59 (driver nuevo, flags viejos),
**antes** de añadir el parámetro de grub → la mejora es del módulo NVIDIA, no de la config.
Decode: sin cambio medible.

---

## 3. Codecs de KV cache (barrido medido)

ctx 98304, mismo resto de flags, 2 reps, prompt frío:

| KV type | VRAM | prefill 31.6k | decode 31.6k | prefill 64k | decode 64k |
|---|---|---|---|---|---|
| **turbo4 (producción)** | 15.6 GB | 1804 | 80.7 | 1675 | **76.7** |
| turbo8 | 17.2 GB | 1796 | 90.3 | 1664 | 75.1 |
| f16 | 19.9 GB | 1827 | 85.5 | 1703 | 73.7 |
| q8_0 | 17.3 GB | 1817 | 69.7 | 1691 | **56.1** (−27 %) |

**Conclusiones**
- `q8_0` es el peor a contexto profundo (dequant más caro que turbo4 y que f16). **No usar.**
- `f16` empata con turbo4 en velocidad y cuesta +4.3 GB → **no cabe a 264K** (8 GiB de KV).
- turbo8 no compensa (+1.6 GB, empata a 64k).
- El prefill no depende del codec (es de pesos).

### VBR (`-ct vbr`) a 264K — medido

| KV | prefill | decode | VRAM pico |
|---|---|---|---|
| turbo4 | 1272 | 57.2 / 58.0 | 19.8 GB |
| `-ct vbr` (entry f16, floor 1.25) | 1286 / 1272 | **50.1 / 49.4** | 23.0 GB |

**VBR pierde ~14 % de decode y +3.2 GB** con este uso. Es la compra "más calidad de KV a
cambio de velocidad y VRAM"; no entra en producción. Si se quiere VBR por calidad, usar el
floor implícito t4 (sin `-ct vbr` explícito), que se comporta casi como turbo4.

---

## 4. A/B de motores: buun vs exllamav3

Mismo modelo, ctx 98304, MTP n=3, KV 4 bits, prompt frío, 2 reps:

| motor | prefill (31.9k) | decode | aceptación draft |
|---|---|---|---|
| buun turbo4 | 1804 | 80.7 | 70 % |
| **exllamav3 (`-cq 4`)** | **2275** (2099/2452) | 91.3 (82/100) | 56 % |

exllamav3 gana **+16…+36 % prefill** y ~10 % decode **con menos aceptación de draft**
(su verificador es más rápido). Aviso: esto se midió con el servidor fino del repo
(`python -m exllamav3.server`), **no con tabbyAPI**; tabbyAPI añade batching continuo,
paged KV compartido y prefix cache — relevante para multi-agente, no para el kernel.
Y el exllamav3 instalado **no tiene prefix cache real**, así que para el flujo de agentes
con 264K buun sigue siendo el sistema correcto.

---

## 5. Negativos medidos (no repetir)

| Prueba | Resultado |
|---|---|
| `--threads 8` fijado al CCD con 3D V-Cache (`taskset -c 0-7,16-23`) vs 16 hilos + `--numa distribute` | +0.3 % prefill, decode en ruido → **no aporta** |
| Draft tuning: `--spec-draft-n-max` 2/3/4 × `--spec-draft-p-min` 0.5/0.75/0.9 (ctx 264K, 64k prompt, gen 384, **seed fijo**) | decode 73–78 en todas; aceptación 67–74 %; diferencias dentro del ruido → **dejar n=3 p=0.75** |
| `q8_0` KV | −27 % decode a 64k |
| VBR explícito a 264K | −14 % decode |
| `-otd token_embd.weight=CPU` (doc DFLASH) | sin ganancia |

---

## 6. Referencia externa: `xvus/syv-ai-qwen38-27b-rtx3090`

Repo de otra persona sirviendo **el mismo modelo** con **vLLM** en **una RTX 3090 a 250 W**.
No es comparable 1:1 (otro quant: W4A16/GPTQ, no EXL3), pero marca el techo de lo que se
puede rascar con stack propio:

| métrica (3090, 250 W) | ellos | nosotros (4090, 450 W) |
|---|---|---|
| decode C1, 64k ctx | **120–130 tok/s** (MTP / DFlash2) | 75–88 |
| decode C1, 150k ctx | 96–102 tok/s | ~60–70 |
| prefill 51k | 1200 (W4A16) / 1423 (int8) | 1272 @200k / 1675 @64k |
| follow-up sobre documento de 24k | 1 s (prefix cache) | depende de checkpoints |

**Qué hacen (y por qué les da ~2× decode):**

1. **lm_head y embed_tokens requantizados** — los quants públicos dejan dos matrices bf16
   de 2.5 GB sin tocar; ellos las pasan a int8 (~2.6 GB de vuelta). Cada token lee esos
   pesos: es la palanca directa sobre decode a batch 1.
2. **Drafter barato**: MTP a bf16 cuesta ~3 ms/token de draft (850 MB + lm_head de 248k
   filas); lo requantizan (int8/int4) y usan un **vocabulario de draft de 40k tokens**
   (97.5 % de cobertura de lo que el modelo genera) → 0.5–1 ms por draft y 4 drafts pasan
   a compensar. Solo esto vale ~10 % de throughput single-stream.
3. **split-KV para el verify multi-query**: FA2 solo reparte la secuencia KV entre bloques
   cuando hay **1** query; el verify de MTP tiene 4–5 → en la 3090 usaban 24 de 82 SMs.
   Un kernel Triton de split-KV baja la atención de 57 µs→23 µs (1.5k) y 1.3 ms→120 µs (16k).
4. **int8 prefill (W4A8 Marlin)** — el prefill es compute-bound: +27–30 % (+19 % a 51k).
5. **Estado recurrente GDN en fp16** — 48 de 64 capas son DeltaNet y el config pide fp32
   (~150 MB por request, leído y escrito en cada paso): limita concurrencia, no velocidad
   single-stream (0.15 ms de 17 ms).
6. **Prefix cache para híbridos** (ellos lo activan explícitamente). Nosotros ya usamos
   `--cache-prompt` + `--ctx-checkpoints 24`.
7. **Metodología**: sus números "viejos" de prefill estaban inflados por el prefix cache;
   ahora siembran cada llamada. Mismo error que hay que evitar aquí.

**Transferible a nuestro stack, por orden de ROI esperado:**

| # | Qué | Dónde | Esperado | Estado |
|---|---|---|---|---|
| 1 | **Vocabulario de draft reducido (40k)** | loader + sampler del MTP en buun | −0.8 GB leídos por draft; +15-22 % decode | **pendiente** (ver §10) |
| 2 | `--spec-type draft-mtp,ngram-simple` (drafting desde el contexto) | solo flag | enorme en cargas que citan el prompt | **disponible ya** (ver §10) |
| 3 | **embeddings cuantizados** (hoy BF16 = 2.54 GB) | modelo + loader | 1.3-1.9 GB de VRAM, no velocidad | pendiente |
| 4 | lm_head 6→4 bits | reconversión EXL3 | −0.32 GB por token ≈ +2 % decode | pendiente |
| 5 | int8 prefill | no existe ruta W4A8 en llama.cpp/EXL3 | +20-30 % prefill (trabajo grande) | no portable |
| 6 | Estado GDN fp16 | `src/llama-model.cpp:3835` (`GGML_TYPE_F32` fijo) | VRAM/concurrencia; ~1.7 % a batch 1 | opcional |
| ~~split-KV en el verify~~ | — | — | — | **YA EXISTE** (ver §10) |

---

## 7. Mapa del código del fork (para cualquier parche)

- `ggml/src/ggml-cuda/fattn.cu` — dispatch de atención. Línea ~2360:
  `if (ggml_is_turbo_kv_type(K) || ggml_is_turbo_kv_type(V)) → BEST_FATTN_KERNEL_VEC`
  (el selector manda turbo a VEC). Línea ~2637: **ruta MMA turbo fusionada** para
  `Q->ne[1] <= 4` y `head_dim` 128/256, gated por `turbo_mma_fused` → nuestro decode
  (1 query, 256) entra ahí. **Trazar con nsys antes de tocar nada.**
- `ggml/src/ggml-cuda/fattn-common.cuh:2080-2182` — `max_blocks_per_sm` + `parallel_blocks`
  y `stream_k` (solo ruta MMA no fusionada; `use_stream_k` se activa para CC >= Ada).
- `ggml/src/ggml-cuda/fattn-mma-turbo.cuh` — kernels MMA para turbo/TCQ.
- `ggml/src/ggml-cuda/mmq-config-ampere.cuh` — tiles MMQ; el cap Ada (`MMQ_X=64` para
  IQ2/IQ3) ya está mergeado en el fork (PR #37) y **no aplica a EXL3**.
- `src/llama-kv-cache.cpp` + `ggml/include/ggml-vbr.h` — controlador VBR.

---

## 8. Cómo medir (sin engañarse)

1. **Para el gateway** (`systemctl --user stop llm-server-router llm-server-manager`) y
   comprueba `nvidia-smi` a 15 MiB antes de arrancar nada. Restaura después.
2. Lanza el binario directo con el comando exacto de `config.yaml`, `CUDA_VISIBLE_DEVICES=0`.
3. Espera `/health` 200 y haz **un warmup descartado** (el primer request compila kernels:
   se han visto 10 s de prompt-eval para 59 tokens).
4. Cada request debe llevar un **nonce distinto** al principio del prompt (si no, el prefix
   cache se come el prefill) y `temperature 0`.
5. Mide con `timings.prompt_n`, `prompt_per_second`, `predicted_n`,
   `predicted_per_second` **del servidor**, no con wall-clock (wall-clock da 15-20 % menos).
6. Para comparar drafts: **mismo `--seed`** en todas las configuraciones; si cambia el texto
   cambia la aceptación y no estás midiendo la config.
7. Dos repeticiones mínimo; la varianza de decode es ±8 % en 128 tokens generados.

Script de cliente: `tools/bench_chat_timings.py` (una petición, un JSON con timings).

```bash
# ejemplo: KV sweep a 98k
for kv in turbo4 q8_0 f16; do
  CUDA_VISIBLE_DEVICES=0 ~/.local/opt/llm-server/buun/bin/llama-server-buun \
    --model /var/llamacpp_models/Qwen3.8-27B-EXL3-3.5bpw --ctx-size 98304 \
    --n-gpu-layers -1 --tensor-split 1 --device CUDA0 --flash-attn on \
    --cache-type-k $kv --cache-type-v $kv --spec-type draft-mtp \
    --spec-draft-n-max 3 --spec-draft-min-p 0.75 --parallel 1 --jinja &
  sleep 15
  uv run python tools/bench_chat_timings.py --port 8080 --prompt-tokens 89000 --gen 192
  kill %1
done
```

---

## 9. Gotchas operativos encontrados

- **`exllama` engine roto sin `transformers`**: devuelve `500 prompt template error: No
  module named 'transformers'`. Se instaló `transformers==5.20.2` en
  `~/.local/opt/llm-server/exllama/venv` para poder medir; habría que añadirlo al instalador.
- **Lock huérfano de torch_extensions**: `~/.cache/torch_extensions/py312_cu128/exllamav3_ext/lock`
  deja `import exllamav3` colgado >8 min. Borrar el lock; arranca en 2.6 s.
- **VBR explícito abre el floor a t1_tcq** (1.25 bpv) aunque el modelo declare floor t4.
- **No editar `config.yaml`**; el ctx de producción es 262144 y el usuario lo quiere al máximo.

---

## 10. Cabeza MTP y portabilidad de `xvus/syv-ai-qwen38-27b-rtx3090`

### 10.1 Qué hay realmente en la cabeza MTP de este checkpoint

Inventario de `Qwen3.8-27B-EXL3-3.5bpw` leyendo las cabeceras de los safetensors
(15.34 GB en total):

| componente | tamaño | notas |
|---|---|---|
| pesos principales | 10.71 GB | 2049 tensores |
| **embed_tokens** | **2.54 GB** | **BF16, sin cuantizar** (16.6 % del fichero) |
| lm_head | 0.95 GB | trellis **6 bits** (`head_bits: 6`) |
| vision tower | 0.92 GB | no se carga sin mmproj |
| **mtp (cabeza MTP)** | **0.21 GB** | **trellis 4 bits** (`mtp_bits: 4`), 39 tensores |

**Conclusión: la cabeza MTP ya está cuantizada a 4 bits (213 MB), no en bf16.** El dato de
`docs/DFLASH-SPEC.md` ("cabeza `blk.64.nextn` ~2,4 GiB bf16") no corresponde a este
checkpoint: 213 MB a 4 bits serían ~850 MB a 16 bits. Pendiente de revisar en ese doc.

Se carga **sin de-cuantizar**: `src/llama-safetensors-qwen35.cpp:566-581` mapea
`nextn.eh_proj`/`nextn.*norm` a `mtp.*` con `quantized_or_plain()`, que devuelve el binding
EXL3 cuando `quant.bind()` acierta (es decir, la trellis se queda en trellis y se de-cuantiza
en kernel).

Por tanto, del punto 5 del repo de referencia ("drafter a int8/int4"), **la mitad ya está
hecha**. Lo que **no** está hecho es la otra mitad: **el vocabulario de draft reducido**.
El draft usa `nextn.shared_head_head` = el `lm_head` completo (0.95 GB, 248.320 filas) en
**cada token borrador**. Con n_max 3 son ~2.9 GB de lectura por ciclo solo para el draft.

Estimación de ganancia si se recorta a ~40k tokens de vocabulario (como ellos: 97.5 % de
cobertura): 0.95 → ~0.15 GB por borrador ⇒ ~0.8 ms menos por draft, ~2.4 ms menos por ciclo.
Con un ciclo de ~13 ms a 64k de contexto eso es **hasta +15-22 % de decode**, además de
habilitar n_max 4-6. Requiere: guardar una cabeza reducida en el modelo (~150 MB) y un mapeo
de ids en la ruta MTP del fork. Esfuerzo medio, es la mejor palanca de drafting que queda.

### 10.2 Portabilidad de las 9 técnicas de syv-ai a buun

| # | Su técnica | ¿Aplica a buun? | Dónde / por qué |
|---|---|---|---|
| 1 | lm_head + embed requantizados | **parcial** | embed sí (2.54 GB BF16 → ~0.6-1.3 GB, solo VRAM); lm_head ya está a 6 bits, bajar a 4 da ~0.32 GB/token ≈ +2 % |
| 2 | patch vLLM para embedding cuantizado | no | llama.cpp ya tiene `token_embd` cuantizado + dequant-on-gather para GGUF; haría falta que el importer EXL3 acepte un embed cuantizado |
| 3 | estado recurrente GDN fp16 | **sí, trivial** | `src/llama-model.cpp:3835-3836`: `recurrent_type_k/v` están fijos a `GGML_TYPE_F32`. Valor: VRAM y concurrencia, ~1.7 % a batch 1 |
| 4 | int8 tensor cores (W4A8 Marlin) | **no** | GGML tiene MMQ int8 para quants GGUF, pero EXL3 des-cuantiza trellis a f16 y usa GEMM f16. No hay ruta de activaciones int8 para trellis |
| 5 | drafter cuantizado + vocab 40k | **mitad hecha** | drafter ya 4 bits (213 MB); **falta el vocab reducido** (§10.1) |
| 6 | split-KV en el verify (FA2 deja SMs ociosos) | **ya existe** | la ruta turbo fusionada llama a `launch_fattn(..., stream_k=true, ...)` y en Ada se activa `use_stream_k` → la KV ya se reparte entre SMs. Su problema era FA2 de vLLM, no llama.cpp |
| 7 | flags afinados + PR de bounds en GDN | n/a | llama.cpp tiene su propia ruta GDN+MTP; vigilar crashes con `crash_bundle` |
| 8 | drafting que lee el contexto (lookup) | **disponible ya** | `--spec-type` acepta **lista separada por comas** y el fork trae `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`, `suffix`, `copyspec`, `recycle` |
| 9 | prefix cache en híbridos | ya | `--cache-prompt` + `--ctx-checkpoints` |

### 10.3 El −32 % de decode a contexto profundo no es un bug de paralelismo

Cálculo con el inventario de KV: 16 capas de atención completa × 4 KV heads × 256 de dim ×
2 (K,V) × 4.125 bits ≈ **1.06 KB por token de contexto**. A 200k de contexto son **~3.4 GB
leídos por paso de decode** (a 64k, ~1.1 GB). A ~1 TB/s eso son ~3.4 ms frente a ~13 ms de
base: explica la mayor parte de la caída 76 → 57 tok/s, y **no se arregla con más
paralelismo** (stream-k ya está). La única palanca es menos bits por elemento de KV
(`turbo3_tcq` = 3.25 bpv, −21 % de bytes de KV) o menos contexto.

### 10.4 Experimentos baratos que salen de aquí

1. `--spec-type draft-mtp,ngram-simple` (y variantes `copyspec`) sobre un prompt que el
   modelo tenga que citar — el equivalente de sus 381 tok/s. **Sin código.**
2. Barrido de KV a 200k de profundidad: `turbo4` vs `turbo3_tcq` vs `turbo2_tcq`
   (menos bytes de KV = decode más rápido a contexto profundo; coste en calidad).
3. Embeddings cuantizados (patch de loader) para recuperar ~1.9 GB de VRAM a 264K.

---

## 11. Captura de tokens generados (base para un draft vocab de 40-60k)

Implementado en `llamacpp_stack/cli/draft_vocab.py` y enganchado en el gateway
(`_cli_impl.py`, ruta streaming y non-stream). **Solo captura**: construir la cabeza
recortada y la tabla de ids es un paso offline aparte.

### Por qué

Con `--spec-type draft-mtp` cada token borrador ejecuta el `lm_head` completo (248.320
filas, 0.95 GB a 6 bits). Una cabeza de 40-60k filas baja ese coste ~6x, que es lo que
separa ~2 de ~3+ tokens aceptados por pasada de pesos. La lista debe cubrir lo que genera
**este** despliegue, no texto web genérico.

### Prueba de viabilidad (medida 2026-10-25)

| comprobación | resultado |
|---|---|
| ¿Un chunk SSE = un token? | sí: 220 chunks → 220 ids al encodear la concatenación |
| `token_to_id(texto_del_chunk)` por chunk | **solo 30 % de aciertos → descartado** |
| encodear el texto acumulado | ~1 µs/token (220 tokens = 0.21 ms) |
| `tokenizers` en el venv del gateway | 0.23.3, ya instalado |
| `tokenizer.json` del modelo | 12.8 MB, presente en el dir del modelo |

Verificación end-to-end: una petición por el gateway (`:11435`) → 1 archivo con
**229 ids distintos / 552 tokens**, y los ids decodifican a tokens reales (`.`, ` It`,
`,`, `2`, `1`, `:`, ` docker`, ` remote`…).

### Configuración (`conf.json` → `logging.draft_vocab`)

| clave | default | rango | qué hace |
|---|---|---|---|
| `enabled` | `false` | — | interruptor maestro |
| `sample_every` | `10` | 1-1000 | 1 de cada N respuestas (1 = todas) |
| `max_tokens_per_response` | `4096` | 64-65536 | tokens que se encoden por respuesta |
| `max_keys` | `200000` | 1000-1e6 | tope de ids distintos por modelo |
| `flush_interval_s` | `600` | 30-86400 | volcado periódico (hilo daemon) |
| `dir` | `""` | — | override del directorio de salida |

Los cambios de configuración se aplican solos en ≤30 s (caché + reconstrucción del
colector, que vuelca lo pendiente antes de recrearse). Código nuevo, no requiere reinicio.

### Salida

```
~/.local/state/llm-server/draft-vocab/<model_id>/counts-YYYY-MM.json
{"model_id": "...", "updated": "...", "month": "2026-10", "tokenizer": "", "counts": {"13": 44, "1049": 15, ...}}
```

Solo **ids y frecuencias**: ni prompts, ni texto generado. Escritura atómica
(tmp+rename) y fusión con el archivo del mes si ya existe. Límites duros en memoria y
disco; todo falla en silencio (una captura rota nunca afecta a la respuesta).

### Siguiente paso (offline)

`tools/build_draft_vocab.py` (pendiente) debe: leer los `counts-*.json`, añadir un núcleo
fijo (dígitos, puntuación JSON, espacios, subpalabras EN/ES, nombres de tools), tomar el
top N, medir cobertura sobre una muestra held-out y emitir tabla `id_draft → token_id` +
cabeza recortada + manifest. El parche de la ruta MTP va después (ver §10).

### Notas de despliegue

El servicio corre una **copia** del paquete en `~/.local/opt/llm-server/python/llamacpp_stack/`
(no editable). Para desplegar un cambio hay que copiar el fichero allí (se dejaron backups
`.bak-dv-*`) y reiniciar `llm-server-router`/`llm-server-manager`.

### 11.1 Diseño de ejecución: 100 % en background

La ruta de respuesta **solo encola** (`queue.put_nowait`, O(1)); la tokenización y la
actualización del histograma ocurren en un hilo worker dedicado, y el volcado en un hilo
flusher separado. Si la cola está llena se descarta la muestra (contador `dropped`) en vez
de bloquear. `observe()` sigue existiendo para uso directo/tests.

Durabilidad frente a reinicios:

| mecanismo | efecto |
|---|---|
| `flush_interval_s` (default **120**) | volcado periódico por hilo daemon; pérdida máxima ≈ 2 min |
| `atexit` → `collector.close()` | volcado en salida limpia del intérprete |
| merge por fichero mensual | un reinicio no borra lo acumulado; se suma al `counts-YYYY-MM.json` |
| conf en `conf.json` | el flag y sus parámetros sobreviven a reinicios |

Verificado: reinicio del gateway a las 19:18 → el fichero conservó **229 ids / 552 tokens**
y la siguiente respuesta añadió **+28 ids / +58 tokens** (257 / 610).

### 11.2 ¿Existe ya un vocabulario reducido en el motor?

El fork trae `--spec-mtp-vocab-size N` ("Qwen-27B MTP public balanced vocabulary; 0
disables, 32768 enables"), pero el código lo aplica **solo a sidecars MTP GGUF externos**
(`tools/server/server-context.cpp:9570`: requiere `speculative.has_dft()` y recorta el GGUF
con `common/mtp-vocab-trim.cpp`). Nuestro MTP es **in-model** (`mtp.*` dentro del EXL3), así
que el flag **no tiene efecto** en nuestra configuración.

### 11.3 Viabilidad del artefacto (layout EXL3 verificado)

El formato EXL3 guarda la matriz como `trellis[in/16, out/16, bits*16]`. Comprobado en dos
tensores del checkpoint:

| tensor | shape real | in | out | bits | in/16, out/16, bits*16 |
|---|---|---|---|---|---|
| `lm_head.trellis` | `[320, 15520, 96]` | 5120 | 248320 | 6 | 320, 15520, 96 ✓ |
| `mtp.fc.trellis` | `[640, 320, 64]` | 10240 | 5120 | 4 | 640, 320, 64 ✓ |

⇒ una cabeza de 40k filas es un **gather de 2500 bloques** en dim1 (**153.6 MB**) más
`svh[:40000]` (80 KB), conservando `suh` y `mul1`. El artefacto es construible sin
de-cuantizar, pero **la inferencia con vocabulario reducido en llama.cpp con nuestro MTP
in-model NO está probada**: requiere el parche de la ruta MTP (mapeo `id_draft ↔ token_id`).

### 11.4 Resultado del A/B real de vocabulario reducido (2026-10-25)

Mismo prompt (64k), seed fija, ctx 262144, GPU1, 2 reps. Drafter = sidecar MTP
`unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf`; el flag `--spec-mtp-vocab-size 32768`
crea un derivado GGUF con cabeza de 32768 filas (log: `[spec] created ... /32768`).

| target | vocab | prefill | decode | aceptación | VRAM |
|---|---|---|---|---|---|
| **EXL3 3.5bpw** | 0 | 1615 | 77.0 / 79.5 | 75 / 77 % | 20.3 GB |
| **EXL3 3.5bpw** | 32768 | 1568 | 79.0 / 80.6 | 72 / 73 % | 24.0 GB |
| GGUF UD-Q4_K_M | 0 | 503 | 1.9 | 71 % | 23.1 GB |
| GGUF UD-Q4_K_M | 32768 | 535 | 2.2 | 71 % | 23.4 GB |

**Conclusiones**

1. **El vocabulario reducido NO paga en EXL3**: +1.5-2 % (ruido) y aceptación 3 puntos peor,
   a cambio de +3.7 GB de VRAM. El trim sí se aplica, así que el resultado es concluyente:
   **el coste del draft no está en el `lm_head`** sino en la sincronización/proceso por paso.
   Queda descartado implementar el parche de la ruta MTP.
2. **GGUF Q4_K_M es ~3x más lento que EXL3 en este fork** para este modelo híbrido
   (prefill 503 vs 1615; decode 1.9-8 vs 77-80) con 23 GB de VRAM. No es una vía viable
   independientemente del flag.
3. El sidecar MTP externo rinde igual que el MTP in-model (77-79.5 vs 74.8-78.8), así que
   es un buen banco de pruebas para futuros drafters sin tocar el modelo EXL3.

La captura (§11) queda **desactivada por defecto** (`logging.draft_vocab.enabled=false`); el
código, los tests y los datos recogidos se conservan y se reactivan con un flag.

### 11.5 DFlash2 vs MTP y "drafter más barato" (2026-10-25)

Mismo target EXL3, prompt de 31.7k y 64.2k, seed fija, `ctx-size-draft` = longitud del prompt
(sin el 500 de §6.5 del DFLASH-SPEC), 2 reps, GPU1.

| prompt | motor | decode (2 reps) | VRAM |
|---|---|---|---|
| 31.7k | **MTP (producción)** | **77.8 / 79.9** | 19.2 GB |
| 31.7k | DFlash2 Q3_K_M (917 MB) | 78.6 / 75.8 | 20.1 GB |
| 31.7k | DFlash2 Q2_K (705 MB) | 79.6 / 72.4 | 19.9 GB |
| 64.2k | **MTP (producción)** | **73.9 / 80.1** | 19.3 GB |
| 64.2k | DFlash2 Q3_K_M | 68.7 / 73.2 | 20.9 GB |
| 64.2k | DFlash2 Q2_K | 67.7 / 70.3 | 20.7 GB |

**Conclusiones**

1. **DFlash2 empata a 31.7k y pierde ~9 % a 64.2k** frente a MTP: su coste crece con la
   profundidad (ver §6.3 del DFLASH-SPEC), y con `ctx-size-draft` dimensionado ya no da 500
   pero sigue perdiendo.
2. **Un drafter más barato no ayuda**: Q2_K (705 MB) rinde igual o peor que Q3_K_M (917 MB) en
   ambos tamaños. Es el mismo patrón que el vocabulario reducido: **el coste no está en los
   bytes del drafter ni en la cabeza, está en la maquinaria de draft+verify**.
3. Con esto quedan cerradas, con datos, todas las palancas de drafting: cabeza MTP ya en 4 bits,
   vocab reducido +2 % (ruido) por +3.7 GB, n_max/p_min planos, DFlash2 pierde a profundidad,
   drafter más pequeño no ayuda, ngram/copyspec roto o sin ganancia.

---

## 12. Comparativa de motores: buun vs tabbyAPI vs vLLM (2026-10-25)

Un solo 4090 (GPU1), mismo prompt sintético, seed fija, cliente **idéntico** de streaming
(`/tmp/llmperf/bench_stream.py`: TTFT y tok/s medidos entre el primer y el último chunk, así
que incluye el overhead HTTP de cada servidor — es lo comparable entre motores).

| motor | modelo | prompt | TTFT | decode |
|---|---|---|---|---|
| **buun** (EXL3 3.5bpw, turbo4, MTP n3) | Qwen3.8-27B-EXL3 | 30.2k | 18.1 s | **74.7** |
| tabbyAPI (exllamav3, cache Q4) | mismo EXL3 | 30.2k | 19.1 s | 53.7 |
| **buun** | mismo EXL3 | 61.1k | 21.6 s | **76.3 / 76.2** |
| tabbyAPI | mismo EXL3 | 61.1k | 15.5 s* | 51.3 / 51.4 |

\* acierto de prefix cache (mismo prefijo que el prompt de 30k).

**tabbyAPI = −29 % decode** frente a buun con los mismos pesos EXL3. Su valor (batching,
paged KV) no se manifiesta a batch 1. Requiere `exllamav3 >= 1.6.2` (el venv de `exllama`
quedó instalado con 1.5.0: para probar tabby hay que subirlo; se revirtió después).

### vLLM 0.32: reconocido pero no arranca en esta máquina

vLLM 0.32 **soporta el modelo de forma nativa** (`Qwen3_5ForConditionalGeneration` en el
registro) y carga el checkpoint INT4 (`RedHatAI/Qwen3.8-27B-INT4`, 18.6 GB). Falla por
cadena de toolchain:

1. El checkpoint declara esquema **FP8 de KV** → vLLM elige un kernel FlashInfer JIT para
   `head_dim 256 / e4m3`, que compila con `nvcc --compress-mode=size` (flag de CUDA 13);
   el nvcc del sistema es **12.4** → `nvcc fatal: Unknown option`.
2. Usando el nvcc CUDA 13 del venv (`site-packages/nvidia/cu13/bin/nvcc`) → *"CUDA compiler
   and CUDA toolkit headers are incompatible"* (mezcla headers 12.4 de `/usr/include`).
3. `--kv-cache-dtype float16` → FA2: *"query and key must have the same dtype"*.
4. `--kv-cache-dtype bfloat16` a 64k → **KV insuficiente** (4.31 GiB necesarios vs 1.99
   disponibles con 18.6 GB de pesos y `gpu-memory-utilization 0.92`).
5. bf16 a 32k → vuelve al JIT de FlashInfer.

**No es una limitación de vLLM ni del modelo**: es el toolkit CUDA. Para desbloquearlo hace
falta instalar un CUDA 13.0 completo (headers + nvcc coherentes) o usar la imagen Docker
`ghcr.io/syv-ai/qwen38-27b-rtx3090`, que ya trae el pin exacto.

Artefactos: modelo INT4 en `/var/llamacpp_models/RedHatAI/Qwen3.8-27B-INT4` (19 GB),
tabbyAPI en `/tmp/tabbyAPI` (sus deps web quedaron en el venv de `exllama`, inofensivas).

### 12.1 vLLM 0.32 desbloqueado (2026-10-25)

El toolkit CUDA 13 **ya estaba instalado** como paquetes pip (`cuda-toolkit 13.0.3.0`,
`nvidia-cuda-nvcc 13.4.92`, headers en `venv/.../nvidia/cu13`). El fallo real era el check
exacto de versión de CCCL (`cuda_toolkit.h`: `CUDART_VERSION` mayor/menor debe coincidir con
`__CUDACC_VER`), y teníamos nvcc 13.**4** con headers 13.**0**. Se desactiva con:

```bash
B=$HOME/.local/opt/llm-server/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH="$B/bin:$PATH" CUDA_HOME="$B" CUDA_PATH="$B"
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export CUDA_VISIBLE_DEVICES=1
vllm serve /var/llamacpp_models/RedHatAI/Qwen3.8-27B-INT4 \
  --served-model-name qwen38-int4 --max-model-len 65536 \
  --gpu-memory-utilization 0.96 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --enable-prefix-caching
```

`0.92` no alcanza (KV 2.3 GiB necesarios vs 1.97 disponibles); con `0.96` quedan 2.97 GiB
de KV y arranca en ~170 s (incluye el JIT de FlashInfer).

### 12.2 Medición con el mismo cliente (32k en frío, 64k con prefix cache)

| motor | prefill 30.2k (frialdad real) | decode 30.2k | decode 61.1k |
|---|---|---|---|
| **buun** (EXL3 3.5bpw, turbo4, MTP n3) | 1670 tok/s (18.1 s) | **74.7** | **76.3 / 76.2** |
| tabbyAPI (exllamav3, Q4) | 1583 tok/s (19.1 s) | 53.7 | 51.3 / 51.4 |
| **vLLM 0.32** (INT4 compressed-tensors) | **3716 tok/s (8.1 s)** | 51.5 | 49.0 / 48.9 |

**Lectura**

1. **vLLM es 2.2× más rápido en prefill** (3.7k vs 1.7k tok/s): sus GEMMs Marlin/int8 sí
   marcan diferencia. Un prompt de 64k en frío pasa de ~38 s a ~17 s.
2. **Decode sin especulación = 49-51 tok/s**, exactamente el techo de ancho de banda
   (18.6 GB de pesos ÷ ~900 GB/s ≈ 48 tok/s). vLLM aquí **no lleva MTP**: con
   `--speculative-config '{"method":"mtp",...}'` y un checkpoint con cabeza MTP
   (`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`) debería escalar ×~2.3 → ~110-115.
3. **vLLM no cabe a 264K en una 4090**: 18.6 GB de pesos + 2.3 GiB de KV fp8 por cada 64k
   ⇒ 264k pide ~9.2 GiB y suma ~27.8 GB > 24.5. Tope real ≈ 96-110k. Para 264K haría falta
   tensor-parallel con las dos 4090 (reparte pesos y KV) — pero entonces no quedan GPUs para
   réplicas.

### 12.3 vLLM + MTP + tensor-parallel: 263K y más rápido a profundidad (2026-10-25)

Checkpoint con cabeza MTP: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (19 GB,
`mtp_num_hidden_layers: 1`).

**En una sola 4090 no cabe con MTP**: pesos 17.7 GiB + cabeza MTP bf16 **2.37 GiB** +
activaciones ~3.6 GiB = 23.7 GiB > 23.6 disponibles → `torch.OutOfMemoryError` al crear el
`ParallelLMHead` del MTP (no es la KV: la KV se asigna después).

**Configuración que sí funciona (y llega a 262144):**

```bash
B=$HOME/.local/opt/llm-server/venv/lib/python3.12/site-packages/nvidia/cu13
PATH="$B/bin:$PATH" CUDA_HOME="$B" CUDA_PATH="$B" \
NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" CUDA_VISIBLE_DEVICES=0,1 \
vllm serve /var/llamacpp_models/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --served-model-name qwen38-mtp --max-model-len 262144 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --enable-prefix-caching --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Arranque ~176 s, **11.39 GiB de KV**, 23.3 GB en cada GPU.

| prompt | vLLM TP2+MTP prefill | decode | buun (1 GPU) prefill | decode |
|---|---|---|---|---|
| 30.2k | 1923 tok/s (15.7 s) | 74.5 (1er request, kernels fríos) | 1670 | 74.7 |
| 61.1k | 3202 tok/s (19.1 s) | **94.2** | ~1600 | 76.3 |
| 189.9k | 2134 tok/s (89.0 s) | **80.6** | 1272 | 57-58 |
| **241.3k** | 44.3 s (con prefix cache) | **106.1** (64 tokens) | — | — |

**Conclusiones**

1. **263K conseguido con vLLM**, pero **solo con tensor-parallel en las dos 4090**: un solo
   GPU no admite pesos + cabeza MTP + KV de 262k ni con KV fp8.
2. **A profundidad vLLM gana claramente**: +39 % decode a 190k (80.6 vs 57.8), +68 % prefill.
   A 64k, +23 %. A 30k empatan.
3. **Coste**: las dos GPUs quedan ocupadas por una sola instancia ⇒ sin réplicas ni failover,
   y con all-reduce por PCIe (los 4090 no tienen NVLink). Además es **otro quant**
   (GPTQ 4-bit vs EXL3 3.5bpw), así que la calidad no es idéntica.
4. buun sigue siendo la única opción que da 264K **en una GPU**, dejando la otra para réplica.

### 12.4 263K en UNA 4090 con vLLM: dónde está el muro (medido)

vLLM 0.32 trae presets de KV **turboquant** (`turboquant_4bit_nc` 3.8x/+2.7 % PPL,
`k3v4_nc` 3.5x/+10.6 %, `3bit_nc` 4.9x/+20.6 %) y `--language-model-only` (no carga el
vision tower).

**Configuración que arranca a 262144 en una sola GPU** (MTP incluido):

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve <GPTQ-Int4-MTP> --max-model-len 262144 \
  --gpu-memory-utilization 0.98 --max-num-seqs 1 --max-num-batched-tokens 64 \
  --language-model-only --enforce-eager --kv-cache-dtype turboquant_4bit_nc \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

- Con `--max-num-batched-tokens 128` **arranca pero muere en el primer prefill**: el kernel
  Triton de turboquant pide 98 MiB de buffer `mid_o` y solo quedan ~80 MiB.
- Con **64** funciona, pero el prefill se degrada a **375 tok/s** (TTFT 80.7 s para 30k) y el
  decode a **45 tok/s** (por `--enforce-eager` + backend Triton). Peor que buun.

**Presupuesto exacto en una 4090 (23.60 GiB útiles):**

| partida | GiB |
|---|---|
| pesos GPTQ int4 | 11.83 |
| embed_tokens F16 | 2.37 |
| lm_head F16 | 2.37 |
| **lm_head duplicado del MTP (F16)** | **2.37** ← desperdicio puro |
| módulo mtp | 0.79 |
| visión (`--language-model-only`) | 0 (antes 0.86) |
| KV turboquant 4-bit @262144 | 5.18 |
| activaciones (2048 batch + CUDA graphs) | ~2 |

Total con batching de producción: **~26.9 GiB → 3.3 GiB por encima**. De ahí que solo
entre con 64 tokens por chunk y sin graphs.

**La forma de que entre de verdad** (todo medible, nada teórico):

1. **Compartir la cabeza MTP con el `lm_head` principal** (vLLM crea una copia F16 de 2.37 GiB
   porque `tie_word_embeddings=False`; el checkpoint **no** trae `mtp.lm_head`, así que la copia
   es idéntica): −2.37 GiB. Es un parche en `model_executor/models/qwen3_5_mtp.py`.
2. **`embed_tokens` a int8** (2.37 → 1.19): −1.18 GiB. Requiere requant + el wiring del kernel de
   embedding cuantizado (mismo enfoque que `qwen3_5-embed-quant.patch` de syv-ai).
3. Opcional: `lm_head` a int8: −1.18 GiB más.

Con (1)+(2): **23.3 GiB** ⇒ entra a 262144 con CUDA graphs y `--max-num-batched-tokens 2048`.
Con (1)+(2)+(3): ~22.2 GiB, con margen.

Resultado esperado (extrapolado de TP=2 medido, misma GPU por rank): decode ~90 @64k y ~80 @200k
**en una sola 4090 y con 263K**, dejando la otra libre.

### 12.5 Parche vLLM: MTP ya no duplica embed/lm_head (implementado)

**Problema.** Qwen3.5 MTP no tiene embedding ni cabeza propios: `load_eagle_model`
(`vllm/v1/worker/gpu/spec_decode/eagle/utils.py`) apunta el drafter a los módulos del
target **después** de construir el modelo. Pero los constructores de `qwen3_5_mtp.py`
asignaban igualmente un `VocabParallelEmbedding` y un `ParallelLMHead` de 248320×5120 F16
(**2.37 GiB cada uno**) que se tiraban acto seguido: **4.74 GiB de pico transitorio** en el
load. En una 4090 de 24 GB eso es lo que impedía cargar con MTP a contexto completo.

**Parche** (en el repo): `llamacpp_stack/bundle/patches-vllm/vllm-0.32-qwen35-mtp-share-heads.patch`
Aplicar/revertir: `tools/apply_vllm_patch.sh [--revert]` (idempotente; copia `.bak-*` y
`py_compile`). Cambia las dos asignaciones por `PPMissingLayer()`; el sharing que ya existe
en vLLM rellena ambos huecos con los del target.

**Medido**

| | antes | después |
|---|---|---|
| arranque TP=2 @262144 | 176 s | **68 s** |
| OOM al construir el drafter a 0.98 (1 GPU) | sí | **no** |
| salida / inferencia | — | coherente (validado con MTP y con TP=2) |
| decode TP=2 @30.2k | 74.5 | **91.4** |

**Lo que el parche NO arregla.** El estado estacionario es el mismo (los módulos ya se
compartían después del load): pesos 17.36 GiB + KV. Con `turboquant_4bit_nc` a 262144 la KV
pide 5.18 GiB y las CUDA graphs reservan ~2.65 GiB, así que en **una** GPU:

| config 1 GPU | 262144 | prefill | decode |
|---|---|---|---|
| eager + 64 tok/chunk + MTP | sí | **375 tok/s** | 45 |
| eager + 256 tok/chunk + MTP | sí (validado) | ~950 tok/s | 55 |
| graphs + 1024 tok/chunk + MTP | no (KV) | — | — |
| sin MTP + eager + 2048 chunk | sí | ~3700 tok/s | ~50 |

Ninguna bate a buun a 263K (1272 tok/s prefill, 57-58 decode) en una GPU.

### 12.6 Por qué el embed a int8 NO se puede en este checkpoint

El checkpoint es `quant_method: gptq`. `AutoGPTQConfig.get_quant_method()` devuelve `None`
para `VocabParallelEmbedding` (solo cubre lineales y MoE), así que vLLM lo carga sin
cuantizar y **no admite mezclar un segundo método de cuantización** para una capa suelta.
La técnica de syv-ai (embed + lm_head a int8, ~2.4 GiB liberados) funciona porque su
checkpoint es **compressed-tensors W4A16**, donde el grupo del embedding se declara en el
mismo `quantization_config`.

Vías reales, por coste:
1. **TP=2** (medido): 263K, decode 80-106, prefill 2.1-3.2k tok/s. Dos GPUs.
2. **Reconstruir el checkpoint** como compressed-tensors W4A16 + embeddings int8
   (pipeline `llm-compressor`/AutoRound, ~horas + 54 GB de bf16). Después el parche de
   syv-ai encaja y 263K en una GPU quedaría en ~22 GiB **con graphs**.
3. **buun** (hoy): 263K en una GPU, 1272 prefill / 57-58 decode, sin tocar nada.

---

## 13. vLLM en una 4090: tablas de tokens a int8 y matriz final (2026-10-26)

### 13.1 Pipeline implementado (repositorio)

| pieza | qué hace |
|---|---|
| `tools/quantize_token_tables.py` | Requantiza `lm_head` y `embed_tokens` a **int8 group-128** en cualquier checkpoint compressed-tensors W4A16: escribe `weight_packed/weight_scale/weight_shape`, añade los grupos al `quantization_config` y saca los `mtp.*` del `ignore`. Backups `.bak-tokentables`, `--dry-run` con error de round-trip, aborta si > 1 %. |
| `llamacpp_stack/bundle/patches-vllm/vllm-0.32-qwen35-mtp-share-heads.patch` | El MTP deja de allocar copias F16 de embed/lm_head (4.74 GiB de pico transitorio). |
| `llamacpp_stack/bundle/patches-vllm/vllm-0.32-qwen35-embed-quant.patch` | `Qwen3_5Model` pasa `quant_config` a `VocabParallelEmbedding` (sin esto el checkpoint int8 carga la copia densa). |
| `tools/apply_vllm_patch.sh` | Aplica/reverts ambos parches sobre site-packages (idempotente, backups, `py_compile`). **Re-ejecutar tras reinstalar vLLM.** |

Checkpoint preparado y validado: `/var/llamacpp_models/dbirks/Qwen3.8-27B-W4A16-AutoRound`
(AutoRound W4A16 + MTP bf16 → convertido a **lm_head int8, embed int8 y MTP int8**).
Errores de round-trip medidos: lm_head 0.64 %, embed 0.56 %, MTP 0.66-1.53 %.

### 13.2 Medido con el parche + int8 (una 4090, `--language-model-only`, KV turboquant 4-bit, MTP n=3)

| max-model-len | prompt | prefill | decode | estado |
|---|---|---|---|---|
| 262144 | 30.2k | 2734 tok/s | **134.2** | OK |
| 262144 | 61.1k | 3527 tok/s | **95.3** | OK |
| 262144 | 190k | — | — | **OOM** (`turboquant_attn.py: k_full[cached_len]`, 224 MiB) |
| 131072 | 30.2k | 2994 tok/s | **140.8** | OK |
| 131072 | 89.3k | 2860 tok/s | ~73 (24 tok) | OK |

**Comparación directa a igual prompt (mismo cliente de streaming):**

| configuración | 32k decode | 64k decode | 190k decode | 263K |
|---|---|---|---|---|
| buun (EXL3 3.5bpw, 1 GPU) | 74.7 | 76.3 | 57-58 | **sí** |
| vLLM 1 GPU, 131072 | **140.8** | — | — | no |
| vLLM 1 GPU, 262144 | 134.2 | 95.3 | **falla** | no |
| vLLM TP=2, 262144 | 74.5-91.4 | 94.2 | 80.6 | **sí** |

### 13.3 La limitación que queda (y por qué no es configurable)

vLLM dimensiona el pool de KV para **llenar** `gpu_memory_utilization` menos el pico de
activación *perfilado*. El buffer transitorio del prefill largo de este backend
(`turboquant_attn.py`, `k_full[:cached_len]`, ~224 MiB con 190k de contexto) no entra en ese
perfil, así que siempre pide unos MiB más de los que quedan (se han visto fallos por **1.7
MiB**). No hay flag de tope de KV en 0.32 (`--kv-cache-memory` no existe) y el margen entre
"el pool no llega a 5.18 GiB" y "el pool se come todo" es de ~2 centésimas de
`gpu-memory-utilization`. Con `--enforce-eager` el pool crece aún más.

Salidas posibles, en orden de coste:
1. **Usar 131072 en vLLM** (una GPU, 140 tok/s a contexto corto) y **buun para 263K**.
2. Parchear vLLM para reservar un margen fijo (p. ej. 512 MiB) al calcular la memoria
   disponible — ~5 líneas en `v1/worker/gpu_worker.py`, pero es un parche de contabilidad
   interno.
3. TP=2 para 263K con vLLM (medido 80-106 tok/s, ocupa las dos GPUs).

---

## 14. Reducir VRAM con RAM: qué existe y qué sirve aquí (2026-10-26)

### 14.1 Palancas de vLLM investigadas

| palanca | qué hace | ¿sirve para 263K en una GPU? |
|---|---|---|
| `--kv-cache-memory-bytes N` | fija el pool de KV en bytes e **ignora** `gpu_memory_utilization` (descubierto en `CacheConfig`; también existe `--kv-cache-memory`, uso manual) | **sí, y es la que faltaba**: evita que el pool se coma hasta el último MiB |
| `--kv-offloading-size G --kv-offloading-backend native` | `OffloadingConnector`: bloques KV completos a RAM fijada (pinned), DMA asíncrono, amplía el *prefix cache* | no para un pico puntual; ayuda con muchos prefijos largos |
| `--cpu-offload-gb G` | pesos a CPU y streaming por capa en cada forward | no (destroza el decode) |
| `--swap-space G` | swap legacy de preempción | no aplica a una sola petición |
| LMCache / TieringOffloadingSpec | CPU + disco/remoto como niveles de caché KV | no para el pico; sí para reutilización masiva |
| `VLLM_CPU_KVCACHE_SPACE` | solo backend CPU | no |

### 14.2 Medido con pool fijado (`--kv-cache-memory-bytes 5.6 GiB`, max-len 262144)

| prompt | resultado |
|---|---|
| 30.2k | prefill 2791 tok/s, **decode 140.9** ✔ |
| 61.1k | prefill 3548 tok/s, decode ~76 ✔ |
| 89.3k (max-len 131072) | prefill 2860 tok/s ✔ |
| **189.9k** | **OOM** en `turboquant_attn.py: k_full[:cached_len]` pidiendo 326-350 MiB con 326 MiB libres (fragmentación), también con `expandable_segments:True`, sin prefix cache y con el pool capado |

**Conclusión**: el límite no es el pool ni la fragmentación configurable: el backend de atención
turboquant alarga un buffer transitorio proporcional al contexto cacheado en el prefill, y a
~190k necesita más de lo que deja libre el modelo (14.65 GiB de pesos + 2.6 de graphs + 5.6 de
pool). Subir el pool lo empeora; bajarlo rompe el requisito de 262144 (5.18 GiB).

### 14.3 Matriz definitiva (medida)

| escenario | motor | decode | 263K |
|---|---|---|---|
| ≤128k, una GPU | **vLLM int8+MTP** (`--max-model-len 131072`) | **140.9** (32k) | no |
| ≤64k, una GPU | vLLM int8+MTP | 140.9 / 95.3 | no |
| 190k+ en una GPU | **buun** | 57-58 | **sí** |
| 263K, máxima velocidad | vLLM TP=2 | 80-106 | sí |

Recomendación operativa: **vLLM con `--max-model-len 131072` para el día a día** (140 tok/s en
una GPU, deja la otra libre) y **buun para sesiones que necesiten >130k**, o **vLLM TP=2** cuando
haga falta 263K con la máxima velocidad y se acepte ocupar las dos GPUs.

---

## 15. Configuración ganadora: vLLM en UNA 4090 con 262144 (2026-10-26)

> ⚠️ **CORRECCIÓN (§16)**: los números de §13.2 y §15.2 se midieron con
> `--kv-cache-dtype turboquant_4bit_nc`, que en este modelo produce **salida
> degenerada** (verificado con un probe determinista). Son válidos como medida de
> velocidad pero **no como configuración usable**. Los números correctos (KV fp8)
> están en §16.

### 15.1 Comando exacto que funciona

```bash
B=$HOME/.local/opt/llm-server/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH="$B/bin:$PATH" CUDA_HOME="$B" CUDA_PATH="$B"
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=1
vllm serve /var/llamacpp_models/dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --served-model-name qwen38-ar \
  --max-model-len 262144 \
  --kv-cache-memory-bytes 5690831667        # 5.3 GiB: el pool NO se come la VRAM
  --kv-cache-dtype turboquant_4bit_nc       # 4-bit KV (3.8x, +2.7 % PPL)
  --max-num-seqs 1 --max-num-batched-tokens 512 --enable-prefix-caching \
  --language-model-only                     # sin vision tower (14.57 GiB cargados = solo texto)
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Requiere: checkpoint con **lm_head/embed/MTP en int8** (`tools/quantize_token_tables.py` +
`quant_mtp.py`), los dos parches de `patches-vllm/` aplicados, y **quitar `mtp.*` del
`ignore`** (si están cuantizados; el tool lo añade por defecto para checkpoints sin cuantizar).

### 15.2 Medido (una 4090, mismo cliente de streaming)

| prompt | prefill | **decode** | buun (mismo prompt) |
|---|---|---|---|
| 30.2k | 2870 tok/s | **140.8** | 74.7 (**+89 %**) |
| 189.9k | 1427 tok/s | **102.0** | 57-58 (**+77 %**) |
| 241k | — | **falla por 0.3 MiB** | 57-58 (funciona) |

### 15.3 Frontera y por qué

El backend de atención turboquant materializa un buffer fp16 **proporcional al contexto**
durante el prefill (~430 MiB a 241k). Con 14.57 GiB de pesos + 5.25-5.3 GiB de pool, a 241k
quedan 430.31 MiB libres y pide 430.00 ⇒ falla por **0.3 MiB** (fragmentación/granularidad).
Bajando el pool a 5.2 GiB no arranca (el requisito de 262144 es 5.18 + padding).

### 15.4 Estrategias descartadas con datos

| estrategia | resultado |
|---|---|
| **`--language-model-only`** (equivale a "deshabilitar mmproj") | **ya aplicada**: el modelo carga 14.57 GiB (solo texto); sin ella serían ~15.4 |
| **Tabla de tokens a int8** | **aplicada**: lm_head y embed 2.37 → 1.19 GiB cada uno (0.64 % / 0.56 % error) |
| **Tabla de tokens a int4** | **descartada**: error de round-trip **10.2 %** en el embedding (el script aborta solo) |
| **Tabla de tokens a RAM (gather en CPU)** | no soportado por vLLM por módulo; además rompería la captura de CUDA graphs (el gather iría dentro del grafo). Equivalente práctico: cuantizarla (arriba) |
| `--kv-offloading-size` (KV a RAM pinned) | útil para capacidad de *prefix cache*, no para el pico de una petición |
| `--cpu-offload-gb` (pesos a CPU) | destroza el decode |
| pool > 5.3 GiB | el pool se come el margen y el prefill largo OOM antes |

### 15.5 Lo único que falta para 241k-262k en una GPU

Trocear `k_full`/`k_cached_trim` en `vllm/v1/attention/backends/turboquant_attn.py`
(`_continuation_prefill`, línea ~1053): procesar el contexto cacheado por bloques de 32k en
vez de materializarlo entero. ~20-40 líneas. Con eso el pico pasa de 430 MiB a ~60 MiB y
241k-262k entran sin tocar nada más.

**Mientras tanto**: vLLM a `--max-model-len 131072` (140 tok/s, 89k probado) o buun para >200k.

---

## 16. Corrección crítica: el KV turboquant corrompe la salida en este modelo (2026-10-26)

Probe determinista usado en todas las comprobaciones: *"What is 17*23? Answer with the number
only."* → debe terminar en **391**.

| KV cache | salida | veredicto |
|---|---|---|
| **fp16** (defecto) | `...391` | correcta |
| **fp8** | `...391` | correcta |
| **turboquant_4bit_nc** | `... a a a a a a a a` | **corrupta** |
| **turboquant_k3v4_nc** | `... a a a a` | **corrupta** |

Con KV turboquant el decode es rapidísimo (140 tok/s @32k) pero el modelo **delira**. Por
tanto: **ninguna medición de §13.2/§15.2 con turboquant KV es una configuración usable**;
solo valen como cota de velocidad del backend.

### 16.1 Números correctos (KV fp8, salida verificada = 391)

`--max-model-len 131072 --kv-cache-dtype fp8 --gpu-memory-utilization 0.95`, MTP n=3,
tablas int8 + MTP int8, una 4090:

| prompt | prefill | decode | buun (mismo prompt) |
|---|---|---|---|
| 30.2k | 3089 tok/s (9.8 s) | **90.9** | 74.7 (**+22 %**) |
| 61.1k | 4397 tok/s (13.9 s) | **90.2** | 76.3 (**+18 %**) |

### 16.2 262144 con KV fp8 no cabe en una 4090

Requisito del pool: **9.29 GiB** a 262144. Sumando 14.57 GiB de pesos + graphs ≈ 0.2 GiB da
24.06 GiB > 23.60 disponibles → ni arranca con `--max-model-len 262144`. El 4-bit KV sí cabía
(5.18 GiB) pero corrompe. **No hay KV correcto que quepa a 262144 en una sola 4090.**

### 16.3 Parche de troceado: intentado y descartado

Se implementó el troceado de `k_full`/`k_cached_trim` en `turboquant_attn.py`
(dequantización directa en los buffers finales + rotación por bloques de 32k). Se **revirtió**:
solo afecta al backend turboquant, que es el que corrompe la salida, así que no hay forma de
validarlo ni de usarlo. El fichero quedó restaurado; no se añadió parche al repo.

### 16.4 Matriz final corregida

| escenario | motor | decode | 263K |
|---|---|---|---|
| ≤128k, una GPU | **vLLM (KV fp8, int8 tables, MTP)** | **90-91** | no |
| 190k+, una GPU | **buun** | 57-78 | **sí** |
| 263K, máxima velocidad | vLLM TP=2 (fp8 KV) | por medir | sí |
| 263K en una GPU con vLLM | **imposible hoy**: 4-bit cabe pero corrompe; fp8 no cabe | — | no |

---

## 17. 263K en UNA 4090 con vLLM: sí es posible, pero 6× más lento (2026-10-26)

### 17.1 La combinación que entra (y da salida correcta)

`--kv-cache-dtype int4_per_token_head` + `--attention-backend TRITON_ATTN` +
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` + `--max-cudagraph-capture-size 8`.

```bash
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1     # no cobra transitorios como "graph memory"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve <ckpt-int8-tables> --max-model-len 262144 --gpu-memory-utilization 0.95 \
  --max-num-seqs 1 --max-num-batched-tokens 512 --enable-prefix-caching \
  --kv-cache-dtype int4_per_token_head --attention-backend TRITON_ATTN \
  --max-cudagraph-capture-size 8 --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Pool disponible: **6.08 GiB** (el requisito de 262144 con int4 es ~4.6 GiB) → arranca y sirve
**241.324 tokens** de prompt. Verificado con el probe determinista: **391** ✓ (a diferencia de
turboquant, que corrompía).

### 17.2 Precio: el backend Triton dequantiza en cada paso

| prompt | vLLM int4 KV | buun | vLLM fp8 KV |
|---|---|---|---|
| 30.2k | 41.8 | 74.7 | **90.9** |
| 61.1k | 24.4 | 76.3 | **90.2** |
| 189.9k | **9.9** | 57-58 | no cabe |
| 241.3k | **8.4** | ~55 | no cabe |
| 262144 configurado | **sí** | **sí** | no |

La dequantización int4 por token-y-cabeza en Triton cuesta ~10× el decode a contexto profundo.
Es la misma familia de coste que hacía "rápido" a turboquant: allí el camino fusionado se
saltaba trabajo y por eso además corrompía.

### 17.3 Lecciones de memoria (aplicables a cualquier config)

| palanca | efecto medido |
|---|---|
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` | el profiler dejaba de cobrar ~4 GiB de transitorios como "graph memory"; el pool pasó de valores erráticos (4.79-6.45 GiB) a **6.08 estable** |
| `--max-cudagraph-capture-size 8` (en vez de 512) | menos graphs, menos pico; con `--max-num-seqs 1` no se necesitan 512 |
| `--language-model-only` | carga 14.57 GiB: sin vision tower |
| tablas int8 + MTP int8 | −2.8 GiB frente a bf16 |
| `--kv-cache-memory-bytes` | control exacto del pool (necesario con KV de 4 bits) |

### 17.4 Matriz definitiva (una 4090, salida verificada en todas)

| escenario | motor | decode 32k/64k/190k | 263K |
|---|---|---|---|
| ≤128k | **vLLM fp8 KV** | **90.9 / 90.2 / —** | no |
| 190k-262k con velocidad | **buun** | 74.7 / 76.3 / **57-58** | **sí** |
| 262k con vLLM | vLLM int4 KV (Triton) | 41.8 / 24.4 / 9.9 | sí (lento) |

**Recomendación**: vLLM fp8 para ≤128k; **buun para todo lo que pase de 128k**. La vía para
263K "rápido" en vLLM seguiría siendo un checkpoint de ~3-3.5 bpw (pesos 14.57 → ~11.5 GiB) que
deje sitio a fp8 KV (9.29 GiB) — pendiente de probar.

---

## 18. Concurrencia en vLLM: 2 conversaciones a ~97 tok/s en UNA GPU (2026-10-26)

Medido con `--max-num-seqs N`, KV fp8, MTP n=3, tablas int8, `--max-model-len 65536`,
prompts de ~2k y 256 tokens de salida por petición (para que los prefills terminen rápido y
se mida decode concurrente de verdad; con prompts de 30k+ el prefill serializado contamina la
medida).

| secuencias | decode por conversación (mediana) | agregado |
|---|---|---|
| 1 | **99.5** | — |
| 2 | **100.2 / 94.7** | 147 tok/s |
| 4 | **82.0** (71-90) | 205 tok/s |
| 8 | **58.4** (46-75) | **294 tok/s** |

**Por qué**: a batch 1 con MTP el 4090 **no está saturado de ancho de banda** (el paso de
verify deja hueco). Al batchear se amortiza la lectura de pesos: la segunda secuencia cuesta
casi nada por paso. Es lo contrario de lo que asumíamos ("decode = bytes/token, batch no
ayuda"): eso vale sin especulación, no con MTP.

### 18.1 Implicación práctica

| setup | conversaciones | tok/s por conversación | GPUs |
|---|---|---|---|
| buun, réplica por GPU (hoy) | 2 | 57-78 | **2** |
| **vLLM fp8, una instancia** | **2** | **95-100** | **1** |
| vLLM fp8 | 4 | ~82 | 1 |
| vLLM fp8 | 8 | ~58 | 1 |

Es decir: **una sola 4090 con vLLM iguala o supera a las dos réplicas de buun** en conversaciones
simultáneas, y deja la segunda GPU libre (para una segunda instancia, o para buun a 263K).

### 18.2 Matices

- Con contextos largos (30k-100k) cada paso lee además la KV: la concurrencia sigue siendo casi
  gratis en cómputo, pero el decode por conversación baja (medido antes: 90 tok/s a 32k, 90 a
  64k con una secuencia; a 190k no cabe con fp8).
- El límite de 262144 sigue siendo el mismo: fp8 KV no cabe; int4 KV sí pero a 8-10 tok/s; KVarN
  es la vía pendiente.
- El "2 conversaciones a 97" es con KV fp8 y 65536 de ctx. Con 8 secuencias y 65536, el pool de
  KV es 6.05 GiB y entra.

### 18.3 Respuesta a la comparación con réplicas de buun

**1 instancia de vLLM × 2 conversaciones simultáneas, UNA sola 4090** (`--max-num-seqs 2`,
KV fp8, MTP n=3, `--max-model-len 65536`, GPU0, GPU1 libre):

| escenario | conversación 0 | conversación 1 | agregado |
|---|---|---|---|
| prompt 2k (contexto caliente) | **101.7 tok/s** | **95.8 tok/s** | 148 tok/s |
| prompt 30k (prefill en frío) | 86.3 | 30.7* | 25.5 (wall 20.1 s) |
| buun, una réplica aislada (referencia) | 57-78 | — | — |

\* el 30.7 de la segunda conversación es artefacto: su ventana de decode incluye el prefill
chunked de la primera. Con prompts largos en frío el tiempo de turno lo domina el prefill
(17.1 s de TTFT), igual que en buun (18 s a 1.670 tok/s).

**Conclusión**: con contexto caliente o prompt corto —el caso de un agente que reenvía su
historial con prefix cache— **una instancia de vLLM en una GPU da ~96-102 tok/s por
conversación con dos en paralelo, frente a los ~70 de una réplica de buun aislada (+40 %)**.
Con prefill frío de 30k, ambos quedan dominados por el prefill.

**TP=2 para dos conversaciones sale peor** (70.7 / 86.8, mediana 78.8, agregado 108): el
all-reduce por PCIe (los 4090 no tienen NVLink) cuesta más de lo que aporta partir el modelo.
Mejor **una instancia por GPU** que una TP=2: dos instancias independientes dan 2 × ~99 tok/s.

---

## 19. TP=2 + fp8 KV: 263K de verdad, 85-99 tok/s (2026-10-26)

Una instancia con `--tensor-parallel-size 2`, KV **fp8**, `--max-model-len 262144`, MTP n=3,
tablas int8, `--max-num-seqs 4`, ambas GPUs. **Arranca** (12.78 GiB de KV por rango, 22.9 GB por
GPU) y da salida correcta (probe 17×23 = **391** ✓).

| escenario | decode |
|---|---|
| 1 conversación, prompt 2k | **85.2** |
| **1 conversación, prompt 190k** | **98.9** (prefill 123 s → 1547 tok/s) |
| **1 conversación, prompt 241k** | **90.9** (prefill 45 s, prefix cache) |
| 2 conversaciones (2k) | 77.6 de mediana (91.9 / 63.3) |
| 4 conversaciones (2k) | 61.6 de mediana — **agregado 158.5** |
| 8 conversaciones (2k) | 56.4 de mediana — agregado 156.0 |

### 19.1 Límite de conversaciones

- **Máximo tok/s por conversación: 1-2 secuencias** (85 → 78, −9 %). A 4 baja a 62 (−28 %).
- **El agregado satura ~158 tok/s a partir de 4 secuencias** (8 no mejora: 156).
- **A contexto completo (262144) caben ~2 conversaciones**: 25.5 GiB de pool total ÷ 9.29 GiB
  por secuencia llena ≈ 2.7. Para más concurrencia hay que bajar `--max-model-len` o el KV.

### 19.2 Comparación con lo que había

| setup | 1 conversación | 2 conversaciones | 263K |
|---|---|---|---|
| buun, 1 GPU @190k | 57-58 | — | sí |
| vLLM 1 GPU, fp8 @131072 | 99.5 | 96-102 | **no cabe** |
| **vLLM TP=2, fp8 @262144** | **85-99** | **78** | **sí** |
| vLLM TP=2, fp8 @65536 | 50.7 | 78.8 | no (no aplica) |

**TP=2 con fp8 KV es la primera configuración que da 263K real, salida correcta y ~90-99 tok/s
por conversación** — un +70 % sobre buun a 190k, con prefill 1.5-3k tok/s.

Curiosidad a verificar: TP=2 con `--max-model-len 65536` dio 50.7 tok/s a una secuencia, pero
con 262144 dio 85.2. La diferencia no está explicada (posible efecto de geometría de bloques o
de la medición tras el arranque); conviene repetir antes de sacar conclusiones.

---

## 20. vLLM TP=2 configurado en producción como `Nombre [vLLM]` (2026-10-26)

### 20.1 Qué se registró

Entrada de catálogo (`~/.local/state/llm-server/catalog.json`, id **`Qwen3.8-27B-AutoRound-INT4-[vLLM]`**):

```json
{
  "model_id": "Qwen3.8-27B-AutoRound-INT4-[vLLM]",
  "local_path": "/var/llamacpp_models/dbirks/Qwen3.8-27B-W4A16-AutoRound",
  "backend": "vllm",
  "ctx_size": 262144,
  "tensor_split": "1,1",
  "aliases": ["qwen3.8-27b-vllm", "qwen3.8-262k-[vLLM]"],
  "server_overrides": {
    "replicas": 0,
    "vllm": {
      "max_model_len": 262144, "tensor_parallel_size": 2, "max_num_seqs": 2,
      "max_num_batched_tokens": 1024, "kv_cache_dtype": "fp8",
      "gpu_memory_utilization": 0.90, "language_model_only": true,
      "enable_prefix_caching": true, "max_cudagraph_capture_size": 16,
      "trust_remote_code": true,
      "speculative_config": {"method": "mtp", "num_speculative_tokens": 3}
    }
  }
}
```

Variables en `~/.config/llm-server/llm-server.env` (las hereda el wrapper `~/.local/bin/vllm-server`):

```
CUDA_HOME=<venv>/lib/python3.12/site-packages/nvidia/cu13
CUDA_PATH=<mismo>
NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
```

El renderer genera el comando con quoting correcto y `--served-model-name 'Qwen3.8-27B-AutoRound-INT4-[vLLM]'`
(`llamacpp_stack/cli/server_commands.py:build_vllm_server_command`, `replica.py:shell_quote`).

### 20.2 Verificado

- `GET /v1/models` (llama-swap :11436) lista:
  - `Qwen3.8-27B-AutoRound-INT4-[vLLM]`
  - `qwen3.8-262k-[vLLM]` (alias)
  - `qwen3.8-27b-vllm` (alias)
- Petición real por el gateway (:11435): carga vLLM TP=2 (~2,5 min con boot + primera petición),
  responde **391** ✓ y devuelve `"model": "Qwen3.8-27B-AutoRound-INT4-[vLLM]"`.

### 20.3 Coste de conmutación (importante)

vLLM TP=2 ocupa **las dos GPUs**, así que no puede convivir con los modelos buun. llama-swap
carga bajo demanda y, tras servir la petición, el manager vuelve a cargar el modelo base de
buun (observado en el journal). **Cada cambio de motor cuesta ~2-3 min.**

Recomendación: elegir uno como motor por defecto.
- `[vLLM]` si la prioridad es 263K + 85-99 tok/s (y ~2 conversaciones llenas).
- buun EXL3 si la prioridad es arranque rápido y réplicas (comparte GPUs, 57-78 tok/s).

---

## 21. Modo mantenimiento para pruebas (2026-10-26)

Para que el tráfico de clientes no contamine las medidas, el gateway tiene un flag runtime:

```bash
touch ~/.local/state/llm-server/maintenance     # activa
rm    ~/.local/state/llm-server/maintenance     # desactiva
```

Comportamiento mientras el flag existe (`_cli_impl.py`, `Handler._maintenance_gate`):

| cliente | respuesta |
|---|---|
| loopback (127.0.0.1 / ::1) | **200** normal |
| no loopback (LAN, etc.) | **503** `{"error": {"message": "maintenance mode: only local requests are accepted", "code": "maintenance"}}` + `Retry-After: 60` |

Verificado: con el flag, `http://127.0.0.1:11435/v1/models` → 200 y `http://192.168.200.133:11435/v1/models`
→ 503; sin el flag, la IP LAN vuelve a su comportamiento normal (308 al redirect HTTPS).

**Protocolo de pruebas** (a partir de ahora): activar el flag → parar/levantar motores y medir
por loopback → desactivar. Si además se para el router/manager para dejar las GPUs libres, los
clientes ven conexión rechazada; con el flag ven 503 limpio y el estado se recupera sin reiniciar
el gateway.

Nota: el flag solo protege el gateway (`:11435`), que es el puerto que usan los clientes. El
llama-swap interno (`:11436`) sigue expuesto; si se quiere cerrar también, hay que filtrar ese
puerto a nivel de host.

---

## 22. vLLM: métricas en llama-swap y respuesta "vacía" (2026-10-26)

### 22.1 Prefill/Decode en blanco en el dashboard

llama-swap calcula Prefill/Decode **parseando las líneas `slot print_timing` del log del
upstream**. vLLM no las emite nunca (el README ya lo decía en la tabla de métricas por backend),
así que el panel queda vacío para los modelos vLLM.

**Solución implementada**: parche
`llamacpp_stack/bundle/patches-vllm/vllm-0.32-slot-print-timing.patch` — hook en
`vllm/v1/engine/output_processor.py` que, por cada `RequestOutput` terminado (streaming y
no-streaming), emite el par de líneas con el formato exacto de llama.cpp a partir de
`RequestStateStats` (`scheduled_ts`, `first_token_ts`, `last_token_ts`).

Verificado en producción:

```
slot print_timing: id  0 | task    1 | prompt eval time =    87.13 ms /    67 tokens (  1.30 ms per token,   768.94 tokens per second)
slot print_timing: id  0 | task    1 |        eval time =   482.47 ms /    52 tokens (  9.28 ms per token,   107.78 tokens per second)
```

Se aplica/reverts con `tools/apply_vllm_patch.sh` (idempotente, backups, `py_compile`).

### 22.2 "No hay respuesta": era el thinking, no un fallo

Sin `--reasoning-parser`, el modelo devuelve **su razonamiento como `content`**; con
`max_tokens` corto la respuesta nunca llega al resultado final y parece que no contesta
(ej. `We need answer user's request: ...` a medias).

**Solución**: `reasoning_parser: qwen3` en `server_overrides.vllm` del catálogo
(el renderer emite `--reasoning-parser qwen3`). Tras el cambio:

```
content: "\n\n391"          ← respuesta final limpia
```

### 22.3 Nota operativa

vLLM TP=2 ocupa las dos GPUs: cada cambio de motor cuesta ~65 s de carga. Si un cliente tiene
timeout de 60 s, la primera petición tras un cambio parece "sin respuesta". Opciones: subir el
timeout del cliente, o fijar `[vLLM]` como motor por defecto (y buun bajo demanda).

---

## 23. "No hay respuesta" en vLLM con clientes de agente: faltaban los flags de tools (2026-10-26)

Síntoma: los clientes (OpenCode y similares, que **siempre envían `tools`**) no obtenían
respuesta. El API devolvía **HTTP 400** en 4 ms:

```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

No era el modelo ni la carga: vLLM rechaza cualquier petición con `tools` si no se configuran
el parser de tool-calls. El gateway lo envolvían como `502 upstream unavailable: HTTP 400`.

**Fix** (`server_overrides.vllm` del catálogo, renderizado por `build_vllm_server_command`):

```json
"enable_auto_tool_choice": true,
"tool_call_parser": "qwen3_xml"
```

`qwen3_xml` es el formato de tool-calls de la familia Qwen3.5/3.8 (el mismo que usan las
recetas de syv-ai y allanchan).

**Verificado tras el cambio**

| prueba | resultado |
|---|---|
| tools + streaming (primera, carga en frío) | **200**, 64 s (≈60 s de carga), `tool_calls` en el SSE |
| tools + streaming (segunda, modelo cargado) | **200**, **1,64 s**, `tool: list_files`, `args: {"path": "/tmp"}` |
| `slot print_timing` | prefill 1.494 tok/s, decode 127,8 tok/s |

### 23.1 Pendiente opcional: `sendLoadingState`

`config.yaml` se renderiza con `sendLoadingState: false` (hardcoded en `replica.py:1055` y
`_cli_impl.py:3881`). Con vLLM TP=2 el primer request tras un cambio de motor tarda ~60-65 s y
el cliente **no recibe ningún evento de progreso**, lo que también se percibe como "no responde".
Si se quiere, se puede poner a `true` en el renderer, pero conviene probarlo antes con los
clientes reales (algunos no manejan bien los eventos de carga).

### 23.2 CORRECCIÓN: el dashboard no lee el log, lee `timings` de la respuesta

El parche de `slot print_timing` (arriba) sirve para `llm-server logs`, pero **no** es de donde
llama-swap saca Prefill/Decode. El router los lee del objeto **`timings` de la respuesta**
(`/api/metrics/activity` → `prompt_per_second`, `tokens_per_second`), igual que `llama-server`.
vLLM solo devuelve `usage`, por eso el panel seguía vacío.

Parche nuevo: `patches-vllm/vllm-0.32-response-timings.patch`
- añade `timings: dict | None` a `ChatCompletionResponse` y `ChatCompletionStreamResponse`;
- lo rellena desde `RequestStateStats` (`scheduled_ts`, `first_token_ts`, `last_token_ts`) con
  los 8 campos de llama.cpp (`prompt_n/ms/per_token_ms/per_second`, `predicted_*`).

**Verificado** (respuesta no-stream y último chunk de streaming) y en
`GET :11436/api/metrics/activity`:

```json
{"model": "Qwen3.8-27B-AutoRound-INT4-[vLLM]", "tokens": {
  "input_tokens": 63, "output_tokens": 49,
  "prompt_per_second": 751.04, "tokens_per_second": 130.69}}
{"model": "Qwen3.8-27B-AutoRound-INT4-[vLLM]", "tokens": {
  "input_tokens": 55, "output_tokens": 40,
  "prompt_per_second": 725.13, "tokens_per_second": 95.50}}
```

Con esto el dashboard tiene Prefill/Decode para vLLM (refrescar con Ctrl+F5). Los parches vLLM
se gestionan juntos con `tools/apply_vllm_patch.sh`.

### 23.3 Los ids de modelo no pueden llevar espacios

El id `Qwen3.8-27B-AutoRound-INT4 [vLLM]` funcionaba con curl pero **no es un id válido** para
clientes/SDK (se rompe al parsear, al construir URLs o en la validación del cliente). Renombrado
a **`Qwen3.8-27B-AutoRound-INT4-[vLLM]`** (guion antes del sufijo, sin espacios) y el alias
`qwen3.8-262k [vLLM]` → `qwen3.8-262k-[vLLM]`. Verificado: `/v1/models` sin espacios, petición
200 con `"model": "Qwen3.8-27B-AutoRound-INT4-[vLLM]"`, y `timings` (140 tok/s decode / 875
prefill) + actividad registrada.

---

## 24. Visión activada en el modelo vLLM: coste medido (2026-10-26)

Se quitó `--language-model-only` del modelo `Qwen3.8-27B-AutoRound-INT4-[vLLM]` y se añadió
`limit_mm_per_prompt: {image: 2, video: 0}`. El checkpoint ya traía el vision tower
(`model.visual.*`, 333 tensores, ~0,86 GiB, fuera de cuantizar por el `ignore` del
compressed-tensors).

**Verificado con OCR**: imagen sintética 640×240 con el texto
`VISION TEST / codigo: 391-ALPHA / objeto: triangulo verde` (data URL base64) →

```
content: "\n\nCODIGO=391-ALPHA OBJETO=triangulo verde"   (1,28 s, 236 prompt tokens, 116 de salida)
```

### 24.1 Coste en peticiones de texto (5 reps, 1,4k prompt, 256 generados, timings del servidor)

| | prefill | decode (mediana) | rango decode |
|---|---|---|---|
| visión **OFF** (`--language-model-only`) | 1974 tok/s | **96,5** | 89,5 – 110,8 |
| visión **ON** | 1980 tok/s | **92,0** | 90,3 – 104,4 |

- **Prefill: idéntico** (dentro del ruido).
- **Decode: ~5 % más lento** (96,5 → 92,0). Los rangos se solapan, así que parte es ruido, pero
  la mediana baja de forma consistente en las dos tandas.
- **VRAM**: pesos 7,48 → **7,94 GiB por rango** (+0,46), y el pool de KV baja de 12,78 a
  **11,58 GiB por rango**. Sigue habiendo de sobra para 262144 (necesita 4,65 GiB/rango).
- El encoder visual **no se ejecuta** en peticiones de texto; el coste viene del wrapper
  multimodal (merge de embeddings por paso) y del menor pool de KV.

### 24.2 Conclusión

Si no vas a mandar imágenes, `--language-model-only` es ~5 % más rápido en decode. Si las
necesitas (capturas, OCR, diagramas), el sobrecoste es pequeño y solo en decode de texto; las
peticiones con imagen funcionan y leen texto correctamente.

---

## 25. "Los clientes no procesan imágenes": el gate de mmproj del gateway (2026-10-26)

Síntoma: las peticiones con imagen a `Qwen3.8-27B-AutoRound-INT4-[vLLM]` **por el gateway**
(:11435) devolvían 400 aunque el motor vLLM tuviera visión activada y funcionase por llama-swap
(:11436):

```
model 'Qwen3.8-27B-AutoRound-INT4-[vLLM]' is installed without multimodal projector
support (mmproj). Use a vision model or re-add/update this model ...
```

**Causa**: el gateway aplica un gate heredado de llama.cpp — solo deja pasar imágenes si el
modelo tiene `mmproj_path` de un GGUF (`_has_configured_mmproj_runtime`). vLLM **no usa mmproj**:
el vision tower va dentro del checkpoint y vLLM lo carga solo. Por eso el motor lo procesaba
perfectamente pero el gateway rechazaba la petición antes de llegar a él (y mis pruebas directas
contra :11436 sí funcionaban, lo que despistó el diagnóstico).

**Fix** (`_cli_impl.py`): helper `_model_is_vllm_multimodal(model)` (backend `vllm` ⇒
multimodal nativo) y se usa en:
- el gate de imágenes de `/v1/chat/completions` y de `/api/chat` (ollama),
- el flag de capacidad `vision`,
- `allow_tool_output_images` (imágenes devueltas por tools).

**Verificado por el gateway** con la imagen sintética:

```
content: "\n\nCODIGO=391-ALPHA OBJETO=triangulo verde"
prompt_tokens: 236        ← incluye los tokens de imagen
```

También documentado en §24: activar visión cuesta ~5 % de decode en texto y +0,46 GiB/rango;
el encoder solo se ejecuta cuando hay imagen.
