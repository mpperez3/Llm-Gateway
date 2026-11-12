---
name: llm-server-installer-update
description: Instalación, actualización y migración de LLM Server (user↔system) sin perder configuración. Úsala SIEMPRE que haya que ejecutar `llm-server install`, actualizar un componente (llama.cpp, llama-swap, buun, vLLM), migrar una instalación existente, añadir una opción nueva al menú del instalador o recuperar una instalación que quedó a medias. Incluye las reglas de idempotencia y los errores de migración que el usuario prohibió repetir.
---

# Instalador y actualizaciones — LLM Server

## 0. Regla de oro

El instalador **no destruye estado**. Si un cambio hace que se pierda `api_key`, `idle_ttl`,
`models-dir`, certs, puertos o modelos del catálogo, el cambio está mal, aunque "funcione".

Errores que ya ocurrieron y no deben repetirse (2026-09-16): no portar la carpeta de modelos, no
portar certs, perder el token de API, no portar binarios compilados, detectar mal el modo y
pedir `sudo` en una instalación user.

## 1. Antes de ejecutar

```bash
uv run llm-server info
uv run llm-server config-keys --format json | jq '.idle_ttl, .api_port, .api_auth'
ls -la ~/.local/state/llm-server/ 2>/dev/null
ls -la /etc/llm-server/ /var/lib/llm-server/ 2>/dev/null   # si hay instalación system
```

- ¿User, system o ambas? Si hay las dos, hay que decidir explícitamente: migrar, eliminar la
  que no toca, o cancelar. Nunca dejar dos conviviendo.
- ¿Hay modelos? `models-dir` se preserva; si el nuevo modo usa otra ruta, se ofrece hardlink o
  symlink, no re-descarga.

## 2. Flujo obligatorio

1. **`--dry-run` siempre primero.** Lee el plan completo antes de ejecutar.
2. Preguntas agrupadas (Q1–Q8 de `docs/LLM_INSTALL.md`), sin re-preguntar lo ya respondido.
3. `uv run`, nunca `pip`.
4. Componentes ya instalados → opción **skip** (no recompilar).
5. `sudo` se pide **cuando toca** (modo system), no al final ni en modo user.
6. Tras instalar: `config-migrate` → `update` → restart → verificación.

```bash
uv run llm-server install --mode user --backend auto --models-dir /var/llamacpp_models \
  --idle-ttl 300 --dry-run
```

## 3. Actualizar componentes

```bash
uv run llm-server update <model_id> --auto    # un modelo
uv run llm-server update                      # regenera config.yaml
```

- Actualizar un binario no debe obligar a reinstalar todo el stack.
- `config-migrate` añade claves nuevas con defaults y **nunca sobrescribe**; segunda pasada
  debe devolver `changed=False` (idempotencia verificable).
- Si el update falla por socket del manager ausente, mira `RuntimeDirectory` del unit antes de
  tocar código.

## 4. Migración user ↔ system

Checklist mínima (todas obligatorias):

- [ ] `conf.json` migrado con merge (api_key, idle_ttl, puertos, logging, experimental).
- [ ] `catalog.json` migrado **completo**; ningún modelo desaparece.
- [ ] `models-dir` preservado (symlink/hardlink si cambia de ruta).
- [ ] Certs TLS y `llm-server.env` copiados.
- [ ] Binarios compilados (llama.cpp, buun, vLLM, exllama) portados o reutilizados.
- [ ] Units antiguas deshabilitadas; nuevas habilitadas y arrancadas.
- [ ] Backup `*.pre-llm-server-migration-<stamp>` presente.
- [ ] Verificación posterior: `info`, `/v1/models`, `/api/replicas`, una generación real.

Si algo de esto no se puede hacer (p. ej. `sudo`), **para y reporta**; no entregues una
instalación a medias.

## 5. Cambios en el instalador

- Mantén idempotencia: `detect_existing_mode`, `choose_default_swap_port`, preservación de
  rutas y certs.
- Pregunta antes de escribir en disco, abrir puertos o descargar.
- Añade la opción "skip" a cualquier paso que pueda estar ya hecho.
- Si tocas el instalador, **sincronízalo al host remoto** (ver `llm-server-remote-deploy`); el
  usuario no debería tener que pedirlo.
- Documenta el cambio en `docs/LLM_INSTALL.md` si afecta a las preguntas o al orden.

## 6. Verificación final

```bash
uv run llm-server info
curl -s http://127.0.0.1:11435/v1/models | jq -r '.data[].id' | head
curl -s http://127.0.0.1:11435/api/replicas | jq '.[0] // .' | head -20
systemctl --user status llm-server-manager llm-server-router --no-pager | head -30
```

Y una generación real de 8 tokens. `info` en verde no demuestra que el modelo cargue.
