"""Sampled capture of generated token ids for draft-vocabulary tuning.

Why this exists
---------------
With ``--spec-type draft-mtp`` every draft token runs the *full* output head
(248 320 rows, ~0.95 GB at 6 bits). A reduced draft vocabulary (40-60k rows)
makes each draft several times cheaper, which is the difference between ~2 and
~3+ accepted tokens per verified weight pass. But the list has to cover what
*this* deployment actually generates, not generic web text.

This module only *collects*: it accumulates a ``{token_id: count}`` histogram
from a sampled subset of completed responses. Building the head artifact and the
id mapping is a separate offline step (the model file has to be touched).

Feasibility (measured 2026-10-25, Qwen3.8-27B-EXL3-3.5bpw, buun)
----------------------------------------------------------------
* llama.cpp streaming emits exactly one SSE chunk per generated token:
  220 chunks → 220 ids after encoding the concatenation.
* ``Tokenizer.token_to_id(chunk_text)`` is unreliable (30 % hits), so the
  collector encodes the whole accumulated text instead: ~1 µs/token, i.e.
  ~1 ms for a 4k-token response, once, after the response finished.
* ``tokenizers`` is already present in the gateway venv; the model dir ships
  ``tokenizer.json`` (12.8 MB).

Config (``conf.json`` → ``logging.draft_vocab``)
------------------------------------------------
``enabled``                  (bool, default false)  master switch
``sample_every``             (int,  10)  1 of every N completed responses; 1 = all
``max_tokens_per_response``  (int,  4096) tokenize at most this many tokens
``max_keys``                 (int,  200000) cap on distinct ids tracked per model
``flush_interval_s``         (int,  600) write the histogram at most this often
``dir``                      (str,  "")   override output dir

Design rules: fail-open (never break a request), ids only (no text is stored),
atomic writes, bounded memory, one JSON per model per month segment.
"""
from __future__ import annotations

import atexit
import itertools
import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULTS: dict[str, object] = {
    "enabled": False,
    "sample_every": 10,
    "max_tokens_per_response": 4096,
    "max_keys": 200000,
    "flush_interval_s": 120,
    "dir": "",
}

# Hard bounds keep a bad conf.json from turning this into a memory/disk hog.
_BOUNDS = {
    "sample_every": (1, 1000),
    "max_tokens_per_response": (64, 65536),
    "max_keys": (1000, 1000000),
    "flush_interval_s": (30, 86400),
}


def _clamp_int(raw: object, default: int, lo: int, hi: int) -> int:
    try:
        if isinstance(raw, bool):
            raise ValueError("bool not allowed")
        value = int(str(raw).strip()) if isinstance(raw, str) else int(raw)  # type: ignore[arg-type]
    except Exception:
        return default
    return max(lo, min(hi, value))


def normalize_config(raw: object) -> dict[str, object]:
    """Return a fully populated, clamped ``logging.draft_vocab`` block."""
    if not isinstance(raw, dict):
        return dict(DEFAULTS)
    out: dict[str, object] = {}
    out["enabled"] = bool(raw.get("enabled", DEFAULTS["enabled"]))
    for key, (lo, hi) in _BOUNDS.items():
        out[key] = _clamp_int(raw.get(key), int(DEFAULTS[key]), lo, hi)
    directory = raw.get("dir", "")
    out["dir"] = str(directory).strip() if isinstance(directory, str) else ""
    return out


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


class DraftVocabCollector:
    """Thread-safe, bounded ``{token_id: count}`` accumulator."""

    def __init__(self, cfg: dict[str, object], base_dir: Path) -> None:
        self.cfg = normalize_config(cfg)
        self.sample_every = int(self.cfg["sample_every"])
        self.max_tokens = int(self.cfg["max_tokens_per_response"])
        self.max_keys = int(self.cfg["max_keys"])
        self.flush_interval_s = int(self.cfg["flush_interval_s"])
        raw_dir = str(self.cfg.get("dir") or "").strip()
        self.base_dir = Path(raw_dir).expanduser() if raw_dir else Path(base_dir) / "draft-vocab"
        self._lock = threading.Lock()
        self._counter = itertools.count()
        self._counts: dict[str, dict[str, int]] = {}
        self._tokenizers: dict[str, object | None] = {}
        self._dirty = False
        self._last_flush = time.monotonic()
        self._unavailable = False
        self._stats = {"requests": 0, "sampled": 0, "tokens": 0, "flushes": 0, "errors": 0, "dropped": 0}
        self._stop = threading.Event()
        self._flusher: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=2048)
        self._worker: threading.Thread | None = None
        self._start_flusher()
        self._start_worker()

    # ---- lifecycle ------------------------------------------------------
    def _start_flusher(self) -> None:
        """Periodic flush so data is not stranded when traffic goes quiet."""

        def loop() -> None:
            while not self._stop.wait(self.flush_interval_s):
                try:
                    self.flush()
                except Exception:
                    pass

        self._flusher = threading.Thread(target=loop, name="draft-vocab-flush", daemon=True)
        self._flusher.start()

    def _start_worker(self) -> None:
        """Everything expensive happens here, never on the request thread."""

        def loop() -> None:
            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if self._stop.is_set():
                        return
                    continue
                try:
                    if item is None:
                        return
                    model_id, tokenizer_path, texts = item
                    self.observe(model_id, tokenizer_path, texts)
                except Exception:
                    with self._lock:
                        self._stats["errors"] += 1
                finally:
                    self._queue.task_done()

        self._worker = threading.Thread(target=loop, name="draft-vocab-worker", daemon=True)
        self._worker.start()

    def enqueue(self, model_id: str, tokenizer_path: str | None, texts: list[str]) -> bool:
        """Hand a completed response to the worker. O(1), never blocks, never raises."""
        if self._unavailable or not tokenizer_path:
            return False
        try:
            self._queue.put_nowait((model_id, tokenizer_path, texts))
            return True
        except queue.Full:
            with self._lock:
                self._stats["dropped"] += 1
            return False
        except Exception:
            with self._lock:
                self._stats["errors"] += 1
            return False

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        try:
            self.flush()
        except Exception:
            pass

    # ---- sampling -------------------------------------------------------
    def should_sample(self) -> bool:
        """Deterministic 1-in-N decision; the first call always samples."""
        if self._unavailable:
            return False
        with self._lock:
            n = next(self._counter)
        return n % self.sample_every == 0

    # ---- capture --------------------------------------------------------
    def observe(self, model_id: str, tokenizer_path: str | None, texts: list[str]) -> None:
        """Accumulate ids for one completed response. Never raises."""
        if self._unavailable or not tokenizer_path:
            return
        try:
            tokenizer = self._tokenizer(tokenizer_path)
            if tokenizer is None:
                return
            text = "".join(str(t) for t in texts if t)
            if not text.strip():
                return
            # Cheap bound before encoding: ~12 chars/token is a safe upper bound
            # for BPE on mixed code/prose, so we never encode gigabytes.
            text = text[: self.max_tokens * 12]
            ids = tokenizer.encode(text, add_special_tokens=False).ids[: self.max_tokens]
            if not ids:
                return
            with self._lock:
                bucket = self._counts.setdefault(str(model_id), {})
                for token_id in ids:
                    key = str(token_id)
                    if key in bucket:
                        bucket[key] += 1
                    elif len(bucket) < self.max_keys:
                        bucket[key] = 1
                self._stats["sampled"] += 1
                self._stats["tokens"] += len(ids)
                self._dirty = True
                due = (time.monotonic() - self._last_flush) >= self.flush_interval_s
            if due:
                self.flush()
        except Exception:
            with self._lock:
                self._stats["errors"] += 1

    def note_request(self) -> None:
        with self._lock:
            self._stats["requests"] += 1

    def _tokenizer(self, path: str):
        with self._lock:
            if path in self._tokenizers:
                return self._tokenizers[path]
        tokenizer = None
        try:
            from tokenizers import Tokenizer  # noqa: PLC0415 - optional dependency

            tokenizer = Tokenizer.from_file(path)
        except Exception:
            # Missing lib or bad file: disable once, stay silent afterwards.
            with self._lock:
                self._unavailable = True
                self._tokenizers[path] = None
            return None
        with self._lock:
            if len(self._tokenizers) >= 2:  # keep memory bounded across model switches
                self._tokenizers.clear()
            self._tokenizers[path] = tokenizer
        return tokenizer

    # ---- persistence ----------------------------------------------------
    def flush(self) -> None:
        """Atomically write every dirty model histogram."""
        with self._lock:
            if not self._dirty:
                return
            snapshot = {model: dict(counts) for model, counts in self._counts.items() if counts}
            self._dirty = False
            self._last_flush = time.monotonic()
        for model_id, counts in snapshot.items():
            try:
                target_dir = self.base_dir / _safe_name(model_id)
                target_dir.mkdir(parents=True, exist_ok=True)
                path = target_dir / f"counts-{_month()}.json"
                merged = counts
                if path.exists():
                    try:
                        previous = json.loads(path.read_text(encoding="utf-8"))
                        if isinstance(previous, dict) and isinstance(previous.get("counts"), dict):
                            merged = {str(k): int(v) for k, v in previous["counts"].items()}
                            for key, value in counts.items():
                                merged[key] = merged.get(key, 0) + value
                    except Exception:
                        pass
                payload = {
                    "model_id": model_id,
                    "updated": datetime.now(timezone.utc).isoformat(),
                    "month": _month(),
                    "tokenizer": "",
                    "counts": merged,
                }
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)
                with self._lock:
                    self._stats["flushes"] += 1
            except Exception:
                with self._lock:
                    self._stats["errors"] += 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            out = dict(self._stats)
            out["models"] = len(self._counts)
            out["distinct_ids"] = sum(len(v) for v in self._counts.values())
        return out


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(value))
    return cleaned.strip("._") or "model"


_GLOBAL: DraftVocabCollector | None = None
_GLOBAL_KEY: tuple | None = None
_GLOBAL_LOCK = threading.Lock()


def get_collector(cfg: object, base_dir: Path) -> DraftVocabCollector | None:
    """Process-wide collector, rebuilt only when the relevant config changes."""
    global _GLOBAL, _GLOBAL_KEY
    normalized = normalize_config(cfg)
    key = (
        bool(normalized["enabled"]),
        normalized["sample_every"],
        normalized["max_tokens_per_response"],
        normalized["max_keys"],
        normalized["flush_interval_s"],
        normalized["dir"],
        str(base_dir),
    )
    with _GLOBAL_LOCK:
        if _GLOBAL is not None and _GLOBAL_KEY == key:
            return _GLOBAL if normalized["enabled"] else None
        if not normalized["enabled"]:
            if _GLOBAL is not None:
                _GLOBAL.close()
            _GLOBAL, _GLOBAL_KEY = None, key
            return None
        if _GLOBAL is not None:
            _GLOBAL.close()
        _GLOBAL = DraftVocabCollector(normalized, base_dir)
        _GLOBAL_KEY = key
        try:
            atexit.register(_GLOBAL.close)
        except Exception:
            pass
        return _GLOBAL


def reset_global() -> None:
    """Test helper: drop the process-wide collector."""
    global _GLOBAL, _GLOBAL_KEY
    with _GLOBAL_LOCK:
        if _GLOBAL is not None:
            _GLOBAL.close()
        _GLOBAL, _GLOBAL_KEY = None, None
