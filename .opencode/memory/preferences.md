---
memory_file: preferences
version: 1
updated: 2026-11-12
load: always
scope: repo
---
# Preferencias del usuario

Cómo quiere que se trabaje en este repo. Son preferencias, no invariantes: si una choca con
un invariante, gana el invariante.

## Comunicación

- **Conciso por defecto.** Respuestas cortas, sin narrar el proceso. El usuario pide
  "resume, sé conciso" con frecuencia; el detalle va a `docs/` o a la memoria, no al chat.
- **Español** en la conversación; inglés en código, docstrings y mensajes de commit.
- **Sin celebrar ni rellenar.** No abrir con "¡Buena pregunta!" ni cerrar con resúmenes
  redundantes.
- **Admite incógnita.** Prefiere "no lo sé, lo mido" a una hipótesis presentada como hecho.
- **Corrige con dureza** ("mentira", "no es cierto", "no me entero"). No es hostilidad:
  exige verificación. La respuesta correcta es aportar evidencia, no excusas.

## Método de trabajo

- **Medir antes de afirmar.** Cualquier número de rendimiento debe salir de una ejecución
  real en esta máquina, con el comando reproducible a la vista.
- **Terminar la tarea.** No parar a medio camino para preguntar "¿continúo?" cuando ya hay
  instrucciones. ("No vuelvas hasta que tengas un 1 vs 1.")
- **Insistir hasta agotar palancas.** Si una optimización parece imposible, el usuario espera
  que se busque en internet, se lea el código del motor y se pruebe, no que se declare
  imposible.
- **Buscar online cuando lo pide** — y decirlo si no se hizo.
- **Preguntar antes de acciones destructivas o caras**: `sudo`, abrir puertos, borrar
  modelos, descargas de GB, parar servicios.
- **Un cambio por iteración**, con su recarga y su verificación.
- **Guardar tras cada avance aceptado**: snapshot en `configs/history/`, entrada en
  `.opencode/memory/events.md`, y doc si el conocimiento es reutilizable.

## Entorno

- `uv` para todo (ver invariante 29). Nunca `pip`, nunca `venv` puro.
- Linux, Python ≥3.12.
- Host local: 2× RTX 4090 24 GB. Host remoto: `mpperez3@192.168.110.41`, instalación
  user-mode en `~/.local/opt/llm-server`, modelos en `/scratch/tesla8/mpperez3/llm_models`.
- El usuario **no puede** hacer `sudo` en remoto por el agente: haz hasta donde llegues y
  avisa.
- Puertos: `:11435` gateway (aplica budget, dedup, réplicas, tool-repair y escribe
  `api-requests.log`), `:11436` router `llama-swap` (+guard). Los clientes pueden ir
  directos a `:11436` y saltarse el gateway: un fix solo en `:11435` no les llega.

## Criterio técnico

- Prefiere **soluciones generales** (política de placement, familia de modelos) a reglas
  ad-hoc por modelo.
- Prefiere **configuración** sobre código cuando el comportamiento ya es configurable.
- Quiere **paridad entre motores** (llama.cpp / buun / vLLM): mismo `max_tokens`, mismo
  reasoning, mismo formato de stream.
- Desconfía de cifras de terceros: si un repo dice 200 tok/s, espera una explicación de por
  qué aquí no se alcanza, no una excusa.
