---
name: llm-server-remote-deploy
description: Despliegue y verificación de LLM Server en el host GPU remoto (mpperez3@192.168.110.41, sing-mgpu04). Úsala SIEMPRE que haya que probar un cambio "en el otro equipo", sincronizar código, reinstalar o comprobar servicios remotos, o cuando el usuario diga "despliega", "sincroniza", "en remoto", "en 41". Incluye la secuencia segura, los paths reales y las trampas de SSH/instalación que ya han costado horas.
---

# Despliegue remoto — LLM Server

## Datos del host

| Dato | Valor |
|---|---|
| Host | `mpperez3@192.168.110.41` (`sing-mgpu04`) |
| Instalación | user-mode en `~/.local/opt/llm-server` |
| Modelos | `/scratch/tesla8/mpperez3/llm_models` |
| Servicios | `systemctl --user restart llm-server-manager llm-server-router` |
| API | `:11435` gateway, `:11436` router |
| `PATH` | `export PATH="$HOME/.local/bin:$PATH"` en **cada** comando remoto |

El usuario **no da `sudo` al agente** en remoto: haz hasta donde llegues y avisa exactamente
qué comando falta.

## 1. Comprobar antes de tocar nada

```bash
ssh -o ConnectTimeout=10 mpperez3@192.168.110.41 'export PATH="$HOME/.local/bin:$PATH"; llm-server info | head -30'
ssh -o ConnectTimeout=10 mpperez3@192.168.110.41 'curl -s localhost:11435/v1/models | head -c 300'
```

- `ConnectTimeout=10` **siempre**: una sesión SSH vieja puede quedarse colgada y bloquear el
  turno entero.
- Si `llm-server info` falla, no sigas: primero decide si es instalación, `PATH` o servicio.
- No cambies la `api_key` remota. Si hay 401 inesperado, sospecha de una clave rotada.

## 2. Desplegar código

```bash
# opción canónica (script del repo)
LLM_SERVER_REMOTE_HOST=192.168.110.41 \
LLM_SERVER_REMOTE_USER=mpperez3 \
LLM_SERVER_REMOTE_PATH='~/llamacpp-stack' \
bash tools/deploy_to_remote.sh
```

Si necesitas control fino (solo unos ficheros, o el script no encaja):

```bash
tar -czf - -C . llamacpp_stack/install.py llamacpp_stack/cli \
  | ssh -o ConnectTimeout=10 mpperez3@192.168.110.41 \
      'export PATH="$HOME/.local/bin:$PATH"; tar -xzf - -C ~/llamacpp-stack'
```

- Envía **solo lo que cambió**. Un `rsync` del repo entero tarda y puede pisar estado local.
- **Si tocas el instalador, sincronízalo sin que te lo pidan** (es una regla del usuario).
- No subas `.venv`, `node_modules`, logs, ni `catalog.json`/`conf.json` del usuario.

## 3. Reinstalar / actualizar en remoto

```bash
ssh -o ConnectTimeout=10 mpperez3@192.168.110.41 \
  'export PATH="$HOME/.local/bin:$PATH"; cd ~/llamacpp-stack && uv run llm-server install --mode user --backend auto --dry-run'
```

1. `--dry-run` primero y lee el plan.
2. Si el instalador detecta otra instalación, debe ofrecer migrar/eliminar/saltar — si no lo
   hace, es un bug del instalador, no algo que se resuelva a mano.
3. Componentes ya instalados: usa la opción skip. **No recompiles llama.cpp/buun sin motivo**
   ("¿pero qué haces compilando otra vez?").
4. `uv run`, nunca `pip`.

## 4. Verificar de verdad

```bash
ssh -o ConnectTimeout=10 mpperez3@192.168.110.41 '
  export PATH="$HOME/.local/bin:$PATH"
  systemctl --user restart llm-server-manager llm-server-router
  sleep 3
  llm-server info | head -25
  curl -s localhost:11435/v1/models | jq -r ".data[].id" | head
  curl -s localhost:11435/api/replicas | jq ".[0] // ." | head -20
'
```

Y una request real, no solo `/v1/models`:

```bash
ssh -o ConnectTimeout=10 mpperez3@192.168.110.41 'curl -s localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"<MODEL_ID>\",\"messages\":[{\"role\":\"user\",\"content\":\"di ok\"}],\"max_tokens\":8,\"stream\":false}" | head -c 400'
```

Un modelo "cargado" en `/v1/models` no demuestra nada: confirma con una generación y con
`nvidia-smi`.

## 5. Trampas que ya han pasado

| Trampa | Síntoma | Qué hacer |
|---|---|---|
| Sesión SSH de horas | El comando remoto no responde | `ConnectTimeout`, y si sospechas, `ssh -O check` / matar la sesión maestra |
| `PATH` sin `~/.local/bin` | `llm-server: command not found` o 500 `failed to start command` | Exportar `PATH` en el mismo comando |
| Instalación duplicada user+system | El instalador pide `sudo` en modo user | Detectar y eliminar la que no toca; reportar |
| Modelos no portados | El modelo existía en local y desaparece en remoto | Revisar `catalog.json` y `models-dir`; nunca re-descargar sin comprobar antes |
| Certs/token perdidos | 401/HTTPS roto tras migrar | Restaurar desde el backup `*.pre-llm-server-migration-*` |
| Compilar "por si acaso" | Minutos perdidos y binario distinto | Saltar si ya está instalado |
| Desplegar WIP | Reintroduce un bug recién eliminado | Solo se despliega lo verificado en local |

## 6. Reporte

Di qué se desplegó (ficheros/commit), qué se reinició, qué se verificó y **qué no se pudo
hacer por falta de `sudo`**. Nunca cierres con "debería funcionar".
