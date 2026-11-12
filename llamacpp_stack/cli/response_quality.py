"""Detección y registro de respuestas de mala calidad.

Motivación
----------
Un modelo puede devolver respuestas degradadas de forma esporádica: vacías,
cortadas por longitud, derivadas a otro idioma, o fragmentadas en cientos de
líneas de uno o dos caracteres. La tasa es baja (del orden de 1/100), así que no
se reproduce pidiendo lo mismo: hay que **registrar los casos cuando ocurren**,
junto con el prompt y el contexto de ejecución, para poder analizarlos después.

Este módulo no altera la respuesta: solo clasifica y guarda. Fail-open siempre.

Config (``conf.json`` → ``logging.response_quality``)
-----------------------------------------------------
``enabled``                    (bool, true)
``save_cases``                 (bool, true)  guarda prompt+respuesta en disco
``dir``                        (str, "")     override del directorio
``max_saved_bytes``            (int, 2 MiB)  tope del JSON guardado
``min_content_chars``          (int, 20)     mínimo para juzgar idioma
``language_drift_ratio``       (float, 0.15) fracción CJK para marcar deriva
``fragmentation_min_lines``    (int, 30)
``fragmentation_short_chars``  (int, 3)
``fragmentation_ratio``        (float, 0.5)

``LLM_SERVER_RESPONSE_QUALITY=0`` desactiva la monitorización por entorno.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULTS: dict[str, object] = {
    "enabled": True,
    "save_cases": True,
    "dir": "",
    "max_saved_bytes": 2_000_000,
    "min_content_chars": 20,
    "language_drift_ratio": 0.15,
    "fragmentation_min_lines": 30,
    "fragmentation_short_chars": 3,
    "fragmentation_ratio": 0.5,
}

_CACHE: tuple[float, dict[str, object]] | None = None
_TTL_S = 30.0


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


def default_dir() -> Path:
    base = Path.home() / ".local" / "state" / "llm-server"
    env = str(os.environ.get("LLM_SERVER_STATE_DIR") or "").strip()
    if env:
        base = Path(env).expanduser()
    return base / "quality-flags"


def normalize_config(raw: object) -> dict[str, object]:
    out = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return out
    for key in ("enabled", "save_cases"):
        value = raw.get(key)
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                out[key] = True
            elif lowered in {"0", "false", "no", "off"}:
                out[key] = False
    if isinstance(raw.get("dir"), str):
        out["dir"] = raw["dir"].strip()
    for key, low, high in (
        ("max_saved_bytes", 65536, 32 * 1024 * 1024),
        ("min_content_chars", 1, 10000),
        ("fragmentation_min_lines", 5, 5000),
        ("fragmentation_short_chars", 1, 20),
    ):
        try:
            out[key] = max(low, min(high, int(raw.get(key, DEFAULTS[key]))))
        except Exception:
            out[key] = DEFAULTS[key]
    for key in ("language_drift_ratio", "fragmentation_ratio"):
        try:
            out[key] = max(0.0, min(1.0, float(raw.get(key, DEFAULTS[key]))))
        except Exception:
            out[key] = DEFAULTS[key]
    return out


def load_config(*, force: bool = False) -> dict[str, object]:
    global _CACHE
    now = time.monotonic()
    if not force and _CACHE is not None and (now - _CACHE[0]) < _TTL_S:
        return dict(_CACHE[1])
    cfg = dict(DEFAULTS)
    try:
        path = _server_config_path()
        if path is not None:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            logging_block = payload.get("logging") if isinstance(payload, dict) else None
            if isinstance(logging_block, dict):
                cfg = normalize_config(logging_block.get("response_quality"))
    except Exception:
        cfg = dict(DEFAULTS)
    env = str(os.environ.get("LLM_SERVER_RESPONSE_QUALITY") or "").strip().lower()
    if env in {"0", "false", "no", "off"}:
        cfg["enabled"] = False
    elif env in {"1", "true", "yes", "on"}:
        cfg["enabled"] = True
    _CACHE = (now, dict(cfg))
    return dict(cfg)


def _cjk(ch: str) -> bool:
    cp = ord(ch)
    return 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF


def _ascii_letters(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def script_mix(text: str) -> tuple[int, int]:
    """(caracteres CJK, caracteres alfabéticos latinos)."""
    cjk = ascii_letters = 0
    for ch in text:
        if _cjk(ch):
            cjk += 1
        elif _ascii_letters(ch):
            ascii_letters += 1
    return cjk, ascii_letters


def prompt_is_latin(text: str) -> bool:
    cjk, latin = script_mix(text or "")
    return latin >= 20 and cjk <= max(2, latin * 0.02)


def analyze(
    *,
    prompt_text: str,
    content: str,
    reasoning: str,
    finish_reason: str,
    tool_calls: int,
    chunk_sizes: list[int] | None = None,
    cfg: dict[str, object] | None = None,
) -> dict[str, object]:
    """Devuelve flags + métricas de la respuesta."""
    c = cfg or load_config()
    content = content or ""
    reasoning = reasoning or ""
    flags: list[str] = []
    metrics: dict[str, object] = {
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "finish_reason": finish_reason,
        "tool_calls": int(tool_calls or 0),
    }

    if not content.strip() and not tool_calls and (finish_reason or "") in {"stop", ""}:
        flags.append("empty")
    if not content.strip() and reasoning.strip() and not tool_calls:
        flags.append("reasoning_only")
    if (finish_reason or "") == "length":
        flags.append("truncated")

    min_content = int(c["min_content_chars"])
    if len(content) >= min_content:
        cjk, latin = script_mix(content)
        total = cjk + latin
        if total:
            ratio = cjk / total
            metrics["cjk_ratio"] = round(ratio, 3)
            if ratio >= float(c["language_drift_ratio"]) and prompt_is_latin(prompt_text):
                flags.append("language_drift")

        lines = [ln for ln in content.split("\n") if ln.strip()]
        if len(lines) >= int(c["fragmentation_min_lines"]):
            short = sum(1 for ln in lines if len(ln.strip()) <= int(c["fragmentation_short_chars"]))
            ratio_lines = short / len(lines)
            metrics["lines"] = len(lines)
            metrics["short_line_ratio"] = round(ratio_lines, 3)
            if ratio_lines >= float(c["fragmentation_ratio"]):
                flags.append("fragmentation")

    if chunk_sizes:
        tiny = sum(1 for size in chunk_sizes if 0 < size <= 3)
        metrics["chunks"] = len(chunk_sizes)
        metrics["tiny_chunk_ratio"] = round(tiny / len(chunk_sizes), 3)

    metrics["flags"] = flags
    return metrics


def save_case(
    *,
    cfg: dict[str, object],
    payload: dict[str, object],
    prompt_text: str,
    content: str,
    reasoning: str,
) -> str | None:
    """Guarda el caso en disco (fail-open) y devuelve la ruta."""
    if not cfg.get("save_cases", True):
        return None
    try:
        raw_dir = str(cfg.get("dir") or "").strip()
        base = Path(raw_dir).expanduser() if raw_dir else default_dir()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target_dir = base / day
        target_dir.mkdir(parents=True, exist_ok=True)
        request_id = str(payload.get("request_id") or "unknown")
        target = target_dir / f"{request_id}.json"
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "flags": payload.get("flags"),
            "metrics": payload.get("metrics"),
            "model": payload.get("model"),
            "upstream_model": payload.get("upstream_model"),
            "params": payload.get("params"),
            "elapsed_ms": payload.get("elapsed_ms"),
            "stream": payload.get("stream"),
            "prompt_text": prompt_text,
            "content": content,
            "reasoning": reasoning,
        }
        blob = json.dumps(record, ensure_ascii=False)
        limit = int(cfg.get("max_saved_bytes", 2_000_000))
        if len(blob) > limit:
            record["prompt_text"] = prompt_text[-limit // 4 :]
            record["content"] = content[: limit // 4]
            record["reasoning"] = reasoning[: limit // 4]
            blob = json.dumps(record, ensure_ascii=False)
        target.write_text(blob, encoding="utf-8")
        return str(target)
    except Exception:
        return None
