---
memory_file: index
version: 1
updated: 2026-11-12
load: always
scope: repo
---
# Memoria del proyecto — índice

Este directorio es la memoria persistente del proyecto. Sobrevive a sesiones, agentes y
máquinas (está versionada). `AGENTS.md` es el router de entrada; esto es el detalle.

## Qué leer y cuándo

| Fichero | Cuándo leerlo | Coste |
|---|---|---|
| [`invariants.md`](invariants.md) | **Siempre**, antes de proponer o cambiar cualquier cosa | ~2 min |
| [`preferences.md`](preferences.md) | Siempre, al empezar a trabajar | ~1 min |
| [`decisions.md`](decisions.md) | Antes de tocar placement, réplicas, dedup, reasoning, logs o motores | ~3 min |
| [`known-issues.md`](known-issues.md) | **Antes de depurar** cualquiera de los síntomas listados | ~2 min |
| [`events.md`](events.md) | Antes de responder "¿esto ya se probó?" o "¿por qué está así?" | ~3 min |
| [`../skills/`](../skills/) | Antes de ejecutar un workflow que tenga skill (ver índice en `AGENTS.md`) | variable |

Si el trabajo es trivial (typo, renombrar una variable), basta con `invariants.md`.

## Cómo se escribe la memoria

**No es un chat log.** Solo entra lo que seguirá siendo cierto dentro de un mes.

| Tipo de conocimiento | Destino |
|---|---|
| Regla dura del usuario ("nunca…", "siempre…") | `invariants.md` |
| Preferencia de método o estilo | `preferences.md` |
| Decisión técnica con motivo | `decisions.md` |
| Fallo ya cometido / trampa | `known-issues.md` |
| Qué pasó, cuándo, con qué resultado | `events.md` |
| Procedimiento repetible de varios pasos | `.opencode/skills/<nombre>/SKILL.md` |
| Explicación larga de un subsistema | `docs/` (y un puntero desde aquí) |

### Reglas de escritura

1. **Una entrada por hecho.** Si ya existe, actualízala en vez de duplicarla.
2. **Cita la fuente**: fecha, commit, sesión, doc o log. Sin fuente, no entra.
3. **Marca el alcance**: ¿aplica a un modelo concreto o a todo el proyecto? Un invariante
   específico de `qwen3.8-27b-q4_1` debe decirlo.
4. **No copies lo que ya está en `docs/`.** Enlaza.
5. **Actualiza `updated:` del frontmatter** al modificar el fichero.
6. **Lo que caduca se borra o se marca como histórico.** Una memoria con datos viejos es peor
   que no tener memoria.
7. **Nunca guardes secretos**: ni API keys, ni tokens, ni certs, ni rutas privadas de modelos.

### Cuándo escribir

- El usuario expresa una restricción nueva → `invariants.md`.
- Se acepta un cambio de configuración en producción → `events.md` + snapshot en
  `configs/history/`.
- Se descubre una causa raíz → `known-issues.md`.
- Se elige entre alternativas con motivo → `decisions.md`.
- Se repite un procedimiento por tercera vez → crear/actualizar skill.

### Conocimiento en riesgo

`.omo/notepads/` contiene auditorías y mediciones de sesiones pasadas, pero **está gitignored**:
no viaja con el repo y se pierde al cambiar de máquina. Si un dato de ahí sigue siendo cierto y
relevante, extráelo a `docs/` (explicación) o a `events.md`/`decisions.md` (decisión y
medición). No lo dejes solo ahí.

### Higiene

- `events.md` crece; al pasar de 60 entradas, mueve las antiguas a `events-<año>.md`.
- `invariants.md` y `preferences.md` deben caber en ~2 pantallas cada uno. Si crecen, algo es
  una decisión o un known-issue, no un invariante.
- `tests/test_project_memory.py` valida estructura, frontmatter y ausencia de nombres muertos.
  Si falla, la memoria está corrupta: arréglala antes de seguir.
