"""Detección temprana de streams corruptos ("sopa de tokens").

Por qué existe
--------------
Con speculative decoding (drafter externo DFlash2) un paso de decode puede dejar
la secuencia en un estado inconsistente.  El modelo entonces emite una ristra de
tokens válidos pero semánticamente aleatorios: fragmentos de palabra sin
separación, con alfabetos que cambian carácter a carácter (CJK, hebreo, cirílico,
tailandés) mezclados con trozos ASCII.  La generación sigue hasta `stop`: minutos
de GPU quemados y basura para el cliente.

Este detector mira el stream mientras llega y avisa en los primeros cientos de
caracteres para que el gateway pueda cortar la generación.

Señal principal
---------------
Una respuesta legítima cambia de alfabeto en frontera de palabra (con espacio o
puntuación por medio).  El fallo alterna alfabetos *dentro* de la misma palabra:
``斑马ILLED阶回声lehmented`` no tiene un solo separador.  Por eso se mide la
fracción de pares adyacentes de letras cuyo alfabeto cambia, y se exige que la
condición se sostenga en ventanas consecutivas.

Señales secundarias: cuatro o más alfabetos distintos con poco latino, o exceso
de U+FFFD (byte inválido).

Config (``conf.json`` → ``logging.stream_canary``)
--------------------------------------------------
``enabled``               (bool, true)   interruptor maestro
``enforce``               (bool, true)   false = solo registra, no corta
``window_chars``          (int, 400)     ventana deslizante
``min_window_chars``      (int, 120)     mínimo de texto antes de evaluar
``min_mixed_ratio``       (float, 0.12)  pares adyacentes con cambio de alfabeto
``min_scripts``           (int, 4)       alfabetos distintos (señal secundaria)
``max_latin_ratio``       (float, 0.60)  latino máximo para la señal secundaria
``max_replacement_ratio`` (float, 0.03)  fracción máxima de U+FFFD
``confirm_windows``       (int, 2)       ventanas seguidas que deben coincidir

También ``LLM_SERVER_STREAM_CANARY=0`` desactiva el detector por entorno.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from pathlib import Path

DEFAULTS: dict[str, object] = {
    "enabled": True,
    "enforce": True,
    "window_chars": 400,
    "min_window_chars": 120,
    "min_mixed_ratio": 0.10,
    "min_scripts": 4,
    "max_latin_ratio": 0.60,
    "max_replacement_ratio": 0.03,
    "confirm_windows": 2,
}

_CONF_CACHE: tuple[float, dict[str, object]] | None = None
_CONF_TTL_S = 30.0

LATIN = "latin"
SPACE = "space"
REPLACEMENT = "replacement"


def _server_config_path() -> Path | None:
    # LLM_SERVER_CONFIG apunta al config.yaml renderizado: no es JSON, no sirve.
    for name in ("LLM_SERVER_CONF_JSON", "LLM_SERVER_CONF"):
        value = str(os.environ.get(name) or "").strip()
        if value.lower().endswith(".json"):
            return Path(value).expanduser()
    legacy = str(os.environ.get("LLM_SERVER_CONFIG") or "").strip()
    if legacy.lower().endswith(".json"):
        return Path(legacy).expanduser()
    for candidate in (
        Path.home() / ".config" / "llm-server" / "conf.json",
        Path("/etc/llm-server/conf.json"),
    ):
        if candidate.exists():
            return candidate
    return None


def normalize_config(raw: object) -> dict[str, object]:
    """Devuelve una config completa y saneada (idempotente)."""
    out = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return out
    for key in ("enabled", "enforce"):
        value = raw.get(key)
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
            out[key] = True
        elif isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off"}:
            out[key] = False
    for key, low, high in (
        ("window_chars", 80, 4000),
        ("min_window_chars", 40, 2000),
        ("min_scripts", 2, 12),
        ("confirm_windows", 1, 10),
    ):
        try:
            out[key] = max(low, min(high, int(raw.get(key, DEFAULTS[key]))))
        except Exception:
            out[key] = DEFAULTS[key]
    for key in ("min_mixed_ratio", "max_latin_ratio", "max_replacement_ratio"):
        try:
            out[key] = max(0.0, min(1.0, float(raw.get(key, DEFAULTS[key]))))
        except Exception:
            out[key] = DEFAULTS[key]
    return out


def load_config(*, force: bool = False) -> dict[str, object]:
    """Config efectiva con caché de 30 s (fail-open)."""
    global _CONF_CACHE
    now = time.monotonic()
    if not force and _CONF_CACHE is not None and (now - _CONF_CACHE[0]) < _CONF_TTL_S:
        return dict(_CONF_CACHE[1])
    cfg = dict(DEFAULTS)
    try:
        path = _server_config_path()
        if path is not None:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            logging_block = payload.get("logging") if isinstance(payload, dict) else None
            if isinstance(logging_block, dict):
                cfg = normalize_config(logging_block.get("stream_canary"))
    except Exception:
        cfg = dict(DEFAULTS)
    env = str(os.environ.get("LLM_SERVER_STREAM_CANARY") or "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        cfg["enabled"] = False
    elif env in {"1", "true", "yes", "on"}:
        cfg["enabled"] = True
    _CONF_CACHE = (now, dict(cfg))
    return dict(cfg)


def script_of(ch: str) -> str:
    """Clasifica un carácter en un alfabeto grueso."""
    cp = ord(ch)
    if cp < 0x0080:
        return LATIN if ch.isalnum() else SPACE
    if 0x00C0 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF:
        return LATIN
    if cp == 0xFFFD:
        return REPLACEMENT
    if 0x0370 <= cp <= 0x03FF:
        return "greek"
    if 0x0400 <= cp <= 0x052F:
        return "cyrillic"
    if 0x0530 <= cp <= 0x058F:
        return "armenian"
    if 0x0590 <= cp <= 0x05FF:
        return "hebrew"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return "arabic"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    if 0x1100 <= cp <= 0x11FF or 0xAC00 <= cp <= 0xD7AF:
        return "hangul"
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    if 0x30A0 <= cp <= 0x30FF:
        return "katakana"
    if 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF:
        return "han"
    return "other"


def _is_word(script: str) -> bool:
    return script not in {SPACE, REPLACEMENT}


class SoupDetector:
    """Detector incremental con memoria acotada (ventana deslizante)."""

    def __init__(self, cfg: dict[str, object] | None = None) -> None:
        self.cfg = normalize_config(cfg)
        self._window = int(self.cfg["window_chars"])
        self._min_window = int(self.cfg["min_window_chars"])
        self._min_mixed = float(self.cfg["min_mixed_ratio"])
        self._min_scripts = int(self.cfg["min_scripts"])
        self._max_latin = float(self.cfg["max_latin_ratio"])
        self._max_repl = float(self.cfg["max_replacement_ratio"])
        self._confirm = int(self.cfg["confirm_windows"])
        self._ring: deque[str] = deque(maxlen=self._window)
        self._since_eval = 0
        self._confirmations = 0
        self.tripped = False
        self.chars_seen = 0
        self.windows_evaluated = 0
        self.last_stats: dict[str, object] = {}

    def _evaluate(self) -> bool:
        total = len(self._ring)
        if total < self._min_window:
            return False
        classes = [script_of(ch) for ch in self._ring]
        counts = Counter(classes)
        scripts = {c for c in counts if c not in {LATIN, SPACE, REPLACEMENT}}
        latin_ratio = counts.get(LATIN, 0) / total
        repl_ratio = counts.get(REPLACEMENT, 0) / total

        word_pairs = 0
        mixed_pairs = 0
        prev = None
        for cls in classes:
            if not _is_word(cls):
                prev = None
                continue
            if prev is not None:
                word_pairs += 1
                if cls != prev:
                    mixed_pairs += 1
            prev = cls
        mixed_ratio = (mixed_pairs / word_pairs) if word_pairs else 0.0

        self.windows_evaluated += 1
        self.last_stats = {
            "window_chars": total,
            "scripts": sorted(scripts),
            "script_count": len(scripts),
            "latin_ratio": round(latin_ratio, 3),
            "mixed_ratio": round(mixed_ratio, 3),
            "replacement_ratio": round(repl_ratio, 4),
            "chars_seen": self.chars_seen,
            "windows_evaluated": self.windows_evaluated,
        }

        soup = len(scripts) >= self._min_scripts and mixed_ratio >= self._min_mixed
        broken = total >= 200 and repl_ratio >= self._max_repl
        if soup or broken:
            self._confirmations += 1
        else:
            self._confirmations = 0
        return self._confirmations >= self._confirm

    def feed(self, text: str) -> bool:
        """Añade texto; devuelve True la primera vez que confirma el fallo."""
        if self.tripped or not text:
            return False
        for ch in text:
            self._ring.append(ch)
            self.chars_seen += 1
            self._since_eval += 1
            if self._since_eval >= 32:
                self._since_eval = 0
                if self._evaluate():
                    self.tripped = True
                    return True
        if self._since_eval and self._evaluate():
            self.tripped = True
            return True
        return False

    def stats(self) -> dict[str, object]:
        data = dict(self.last_stats)
        data.update({
            "tripped": self.tripped,
            "confirmations": self._confirmations,
            "chars_seen": self.chars_seen,
        })
        return data
