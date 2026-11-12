---
name: llm-server-mmproj-vision
description: Operación de visión (mmproj) en LLM Server, incluido el modo lazy. Úsala SIEMPRE que el usuario pida "activar imágenes", "que procese imágenes", cuando un cliente diga que no hay procesamiento de imagen, cuando haya que decidir entre MTP y mmproj por VRAM, o al depurar por qué una ruta no tiene visión. Contiene la tabla de selección de instancia y las trampas de reescritura de ruta.
---

# Visión y mmproj

## 1. Estado real (no el que crees)

```bash
curl -sk https://127.0.0.1:11435/v1/models -H "Authorization: Bearer $LLM_API_KEY" \
  | jq '.data[] | select(.metadata.mmproj_mode) | {id, vision: .metadata.capabilities.vision, mode: .metadata.mmproj_mode}'
pgrep -af mmproj
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

- La respuesta **autoritativa** es la del gateway (`:11435`): combina el `cmd` de la ruta con el
  estado de carga de llama-swap.
- llama-swap v255 **no devuelve** `capabilities` en `/v1/models` aunque lo acepte en config.
  No concluyas nada mirando solo el router.
- `pgrep`/`nvidia-smi` mandan sobre cualquier metadata: un proceso con `--mmproj` puede
  sobrevivir a la flag que lo creó hasta que expire su TTL.

## 2. Modos

| `mmproj.default_mode` | Comportamiento |
|---|---|
| `always` (default) | Todas las rutas cargan `--mmproj`; cuesta VRAM y tiempo de carga siempre |
| `lazy` | Las rutas se publican **sin** `--mmproj`; se engancha al vuelo cuando llega una imagen |
| `off` | Se elimina `--mmproj`; no hay visión |

Override por modelo: `server_overrides.mmproj_mode`.

## 3. Cómo elige instancia en lazy

**No hay ruta dedicada de visión.** Se usa la base o sus réplicas, y la decisión ocurre
**antes** de la selección de réplica, para que una instancia text-only no reciba la imagen:

| Estado de las candidatas | Elegida |
|---|---|
| Alguna cargada **con** el proyector | Esa, aunque otra esté libre (sin recarga) |
| Todas cargadas, ninguna lo lleva, algunas ociosas | Una **descargada**, para no interrumpir |
| Todas ociosas, una ya lo lleva | Esa, para evitar recarga |
| Todas ociosas, ninguna lo lleva | La que lleve más tiempo ociosa |

La primera fila es la clave: una imagen puntual en una conversación larga no recarga nada, y el
siguiente turno de texto vuelve a la instancia que la conversación seguía.

## 4. TTL y reescritura de ruta

- La ruta elegida se reescribe para añadir `--mmproj` y recibe `ttl = vision_sticky_ttl_s`
  (3600 s por defecto).
- llama-swap aplica un `cmd` reescrito **la siguiente vez que la ruta carga**. Por eso, si la
  instancia ya está viva, se descarga y se espera a que desaparezca antes de enganchar la flag.
- Al expirar el TTL, llama-swap desaloja el proceso. La flag se quita cuando esa instancia
  vuelve a necesitarse **estando descargada**.
- **Nunca reescribas una ruta viva**: reinicia el proceso y tira la KV cache de la conversación.

## 5. Diagnóstico típico

| Síntoma | Causa probable | Acción |
|---|---|---|
| "No procesa imágenes" | Ruta sin `--mmproj` (lazy y no se enganchó) | Verificar `vision` en `/v1/models` del gateway y `pgrep -af mmproj` |
| `503` al mandar imagen | No se pudo enganchar el proyector | Revisar VRAM libre y `mmproj.reload_timeout_s`; reintentar |
| Funciona y al rato no | TTL expirado y ruta desalojada | Es esperado en lazy; subir `vision_sticky_ttl_s` si molesta |
| Imagen ignorada silenciosamente | Se envió a una ruta sin proyector | El gateway no debe permitirlo: es un bug, no config |
| Tras reiniciar unidad sigue `--mmproj` | Proceso residual | Parar units, matar leftovers, arrancar limpio |

## 6. VRAM: MTP antes que mmproj

Con 262K en 24 GiB, MTP y mmproj compiten por el mismo presupuesto. **El usuario prefiere MTP.**
No actives mmproj "por si acaso": activa `lazy` y deja que se cargue cuando haga falta.

## 7. Verificación final

```bash
# 1. petición con imagen real por el camino del cliente
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"<MODEL_ID>","messages":[{"role":"user","content":[
   {"type":"text","text":"describe la imagen en una frase"},
   {"type":"image_url","image_url":{"url":"data:image/png;base64,<...>"}}]}],
 "max_tokens":64,"stream":false}' | jq '.choices[0].message.content'
# 2. confirmar que el proyector está cargado
pgrep -af mmproj
```

Una respuesta de texto plausible **y** el proceso con `--mmproj` es la única prueba válida.
