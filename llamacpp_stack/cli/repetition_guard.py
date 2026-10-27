"""Repetition guard for OpenAI-compatible chat streams.

Long-context generation occasionally degenerates into echoing one fragment (a
word-counting loop, a repeated sentence). The guard watches the streamed text
(reasoning + visible content), aborts the generation when the same window keeps
coming back, retries once with a nudge (optionally with thinking disabled) and
records the incident so the *next* request can tell the model what happened.

Everything is configurable from ``conf.json`` under ``repetition_guard``:

    {
      "enabled": true,
      "window_chars": 120,        # length of the repeated window
      "min_repeats": 3,           # window occurrences that count as a loop
      "min_chars": 600,           # do not judge before this much text
      "buffer_chars": 6000,       # hold streamed text back until this much (or first content)
      "retry": true,
      "max_retries": 1,
      "retry_disable_thinking": true,
      "retry_nudge": "La generación anterior entró en un bucle ...",
      "notify_next": true,
      "notify_ttl_s": 900,
      "notify_count": 1
    }
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

DEFAULTS: dict[str, object] = {
    "enabled": True,
    "window_chars": 120,
    "min_repeats": 3,
    "min_chars": 600,
    "max_count_patterns": 20,
    "buffer_chars": 6000,
    "retry": True,
    "max_retries": 1,
    "retry_disable_thinking": True,
    "retry_nudge": (
        "La generación anterior entró en un bucle de repetición. Responde ahora de forma "
        "directa y breve, sin enumerar ni contar palabras, sin repetir fragmentos y sin "
        "mostrar tu razonamiento paso a paso."
    ),
    "notify_next": True,
    "notify_ttl_s": 900,
    "notify_count": 1,
}


def _conf_path() -> Path:
    return Path(os.environ.get("LLM_SERVER_CONFIG", os.path.expanduser("~/.config/llm-server/conf.json")))


def _state_path() -> Path:
    state = os.environ.get("LLM_SERVER_STATE_DIR", os.path.expanduser("~/.local/state/llm-server"))
    return Path(state) / "repetition_guard.json"


def load_config() -> dict[str, object]:
    cfg = dict(DEFAULTS)
    if os.environ.get("LLM_SERVER_REPETITION_GUARD", "1") == "0":
        cfg["enabled"] = False
        return cfg
    try:
        raw = json.loads(_conf_path().read_text(encoding="utf-8"))
        section = raw.get("repetition_guard") if isinstance(raw, dict) else None
        if isinstance(section, dict):
            cfg.update({k: v for k, v in section.items() if k in DEFAULTS})
    except Exception:
        pass
    return cfg


class RepetitionDetector:
    """Rolling window detector: True once one window repeats min_repeats times."""

    def __init__(self, cfg: dict[str, object]):
        self.window = max(20, int(cfg.get("window_chars") or 120))
        self.min_repeats = max(2, int(cfg.get("min_repeats") or 3))
        self.min_chars = max(self.window, int(cfg.get("min_chars") or 600))
        self.max_count_patterns = int(cfg.get("max_count_patterns") or 20)
        self.buf = ""
        self.raw = ""

    def feed(self, text: str) -> bool:
        if not text:
            return False
        text = str(text)
        self.raw = (self.raw + text)[-8000:]
        # Counting loops change the index every word ("palabra(12)", "palabra(13)"),
        # so compare windows with digits masked out; the raw text also feeds the
        # "word(index)" pattern counter below.
        self.buf += re.sub(r"\d+", "#", text)
        # keep enough history for min_repeats windows plus one
        keep = self.window * (self.min_repeats + 2)
        if len(self.buf) > keep:
            self.buf = self.buf[-keep:]
        if len(self.raw) >= self.min_chars and len(re.findall(r"\w+\(\d+\)", self.raw)) >= self.max_count_patterns:
            return True
        if len(self.buf) < self.min_chars:
            return False
        tail = self.buf[-self.window:]
        if len(tail) < self.window:
            return False
        count = self.buf.count(tail)
        return count >= self.min_repeats


def conversation_key(payload: dict) -> str:
    """Stable key for the request's conversation: hash of its first user message."""
    try:
        for msg in payload.get("messages") or []:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    content = " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
                return hashlib.sha1(str(content or "").encode("utf-8")).hexdigest()[:16]
    except Exception:
        pass
    return ""


def record_incident(model: str, detail: str = "", conv_key: str = "") -> None:
    try:
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "ts": time.time(),
            "model": model,
            "detail": detail[:400],
            "conv_key": conv_key,
            "pending": int(load_config().get("notify_count") or 1),
        }), encoding="utf-8")
    except Exception:
        pass


def pop_pending_note(model: str = "", conv_key: str = "") -> str | None:
    """Return the nudge for the next request (and consume one use)."""
    cfg = load_config()
    if not cfg.get("enabled") or not cfg.get("notify_next"):
        return None
    try:
        p = _state_path()
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        age = time.time() - float(data.get("ts") or 0)
    except Exception:
        age = 1e9
    ttl = float(cfg.get("notify_ttl_s") or 900)
    pending = int(data.get("pending") or 0)
    if model and data.get("model") and str(data.get("model")) != model:
        return None
    if conv_key and data.get("conv_key") and str(data.get("conv_key")) != conv_key:
        return None
    if age > ttl or pending <= 0:
        try:
            p.unlink()
        except Exception:
            pass
        return None
    data["pending"] = pending - 1
    try:
        if data["pending"] <= 0:
            p.unlink()
        else:
            p.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    note = str(cfg.get("retry_nudge") or "").strip()
    if not note:
        return None
    return f"[aviso del gateway] {note}"


def build_retry_payload(payload: dict, cfg: dict[str, object]) -> dict:
    """Copy of the request with thinking disabled (optional) and a nudge appended."""
    retry = json.loads(json.dumps(payload))
    if bool(cfg.get("retry_disable_thinking", True)):
        kwargs = dict(retry.get("chat_template_kwargs") or {})
        kwargs["enable_thinking"] = False
        retry["chat_template_kwargs"] = kwargs
    note = str(cfg.get("retry_nudge") or "").strip()
    if note:
        msgs = list(retry.get("messages") or [])
        msgs.append({"role": "system", "content": f"[aviso del gateway] {note}"})
        retry["messages"] = msgs
    return retry
