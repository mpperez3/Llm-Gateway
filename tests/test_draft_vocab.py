"""Tests for the sampled draft-vocabulary collector (logging.draft_vocab)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llamacpp_stack.cli import draft_vocab


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _StubTokenizer:
    """Maps each whitespace-separated word to a stable id."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        self.calls.append(text)
        return _Encoding([abs(hash(word)) % 1000 for word in text.split()])


def _collector(tmp_path: Path, **overrides) -> draft_vocab.DraftVocabCollector:
    cfg = {"enabled": True, **overrides}
    return draft_vocab.DraftVocabCollector(draft_vocab.normalize_config(cfg), tmp_path)


def _with_tokenizer(collector: draft_vocab.DraftVocabCollector, path: str) -> _StubTokenizer:
    tokenizer = _StubTokenizer()
    collector._tokenizers[path] = tokenizer  # noqa: SLF001 - test seam
    return tokenizer


def test_normalize_clamps_and_defaults() -> None:
    cfg = draft_vocab.normalize_config({"sample_every": 0, "max_keys": 10 ** 12, "flush_interval_s": "x"})
    assert cfg["enabled"] is False
    assert cfg["sample_every"] == 1
    assert cfg["max_keys"] == 1000000
    assert cfg["flush_interval_s"] == draft_vocab.DEFAULTS["flush_interval_s"]
    assert cfg["max_tokens_per_response"] == draft_vocab.DEFAULTS["max_tokens_per_response"]


def test_normalize_rejects_non_dict() -> None:
    assert draft_vocab.normalize_config(None) == draft_vocab.DEFAULTS
    assert draft_vocab.normalize_config([1, 2]) == draft_vocab.DEFAULTS


def test_disabled_config_returns_no_collector(tmp_path: Path) -> None:
    draft_vocab.reset_global()
    assert draft_vocab.get_collector({"enabled": False}, tmp_path) is None


def test_sampling_is_one_in_n_and_first_call_samples(tmp_path: Path) -> None:
    collector = _collector(tmp_path, sample_every=3)
    decisions = [collector.should_sample() for _ in range(7)]
    assert decisions == [True, False, False, True, False, False, True]


def test_observe_accumulates_ids_and_flushes_atomically(tmp_path: Path) -> None:
    collector = _collector(tmp_path, sample_every=1)
    _with_tokenizer(collector, "/models/x/tokenizer.json")
    collector.observe("m1", "/models/x/tokenizer.json", ["hola mundo ", "adios"])
    collector.flush()

    files = list((tmp_path / "draft-vocab" / "m1").glob("counts-*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["model_id"] == "m1"
    assert payload["counts"]
    assert not list(files[0].parent.glob("*.tmp"))


def test_flush_merges_with_existing_month_file(tmp_path: Path) -> None:
    collector = _collector(tmp_path, sample_every=1)
    _with_tokenizer(collector, "/t.json")
    collector.observe("m1", "/t.json", ["uno dos"])
    collector.flush()

    second = _collector(tmp_path, sample_every=1)
    _with_tokenizer(second, "/t.json")
    second.observe("m1", "/t.json", ["uno dos"])
    second.flush()

    files = list((tmp_path / "draft-vocab" / "m1").glob("counts-*.json"))
    counts = json.loads(files[0].read_text())["counts"]
    assert sum(counts.values()) == 4  # two observations x two tokens


def test_max_keys_bounds_the_histogram(tmp_path: Path) -> None:
    collector = _collector(tmp_path, sample_every=1, max_keys=1000)  # clamped minimum
    _with_tokenizer(collector, "/t.json")
    words = " ".join(f"w{i}" for i in range(1200))
    collector.observe("m1", "/t.json", [words])
    bucket = collector._counts["m1"]  # noqa: SLF001 - test seam
    assert len(bucket) <= 1000


def test_max_tokens_per_response_truncates(tmp_path: Path) -> None:
    collector = _collector(tmp_path, sample_every=1, max_tokens_per_response=64)
    _with_tokenizer(collector, "/t.json")
    collector.observe("m1", "/t.json", [" ".join(f"w{i}" for i in range(500))])
    assert collector.stats()["tokens"] <= 64


def test_missing_tokenizer_is_fail_open(tmp_path: Path) -> None:
    collector = _collector(tmp_path, sample_every=1)
    collector.observe("m1", "/does/not/exist/tokenizer.json", ["hola"])
    collector.flush()
    assert collector.stats()["flushes"] == 0
    assert not list(tmp_path.rglob("counts-*.json"))


def test_observe_without_tokenizer_path_is_noop(tmp_path: Path) -> None:
    collector = _collector(tmp_path, sample_every=1)
    collector.observe("m1", None, ["hola"])
    assert collector.stats()["sampled"] == 0


def test_unsafe_model_name_is_sanitized(tmp_path: Path) -> None:
    collector = _collector(tmp_path, sample_every=1)
    _with_tokenizer(collector, "/t.json")
    collector.observe("../evil/../name", "/t.json", ["uno"])
    collector.flush()
    dirs = [p.name for p in (tmp_path / "draft-vocab").iterdir()]
    assert len(dirs) == 1
    assert "/" not in dirs[0] and not dirs[0].startswith(".")
    # nothing may be written outside the collector's base dir
    assert not (tmp_path.parent / "evil").exists()


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_enqueue_returns_immediately_and_worker_processes(tmp_path: Path) -> None:
    import time

    collector = _collector(tmp_path, sample_every=1)
    _with_tokenizer(collector, "/t.json")
    t0 = time.monotonic()
    assert collector.enqueue("m1", "/t.json", ["uno dos tres"]) is True
    assert time.monotonic() - t0 < 0.05  # never blocks on tokenization
    assert _wait_for(lambda: collector.stats()["sampled"] == 1)
    collector.close()


def test_enqueue_drops_when_queue_is_full(tmp_path: Path) -> None:
    import queue as _queue

    collector = _collector(tmp_path, sample_every=1)
    _with_tokenizer(collector, "/t.json")
    collector._queue = _queue.Queue(maxsize=1)  # noqa: SLF001 - test seam
    collector._queue.put_nowait(("busy", "/t.json", ["x"]))  # noqa: SLF001
    assert collector.enqueue("m1", "/t.json", ["uno"]) is False
    assert collector.stats()["dropped"] == 1
    collector.close()


def test_close_flushes_pending_data(tmp_path: Path) -> None:
    collector = _collector(tmp_path, sample_every=1, flush_interval_s=3600)
    _with_tokenizer(collector, "/t.json")
    assert collector.enqueue("m1", "/t.json", ["uno dos"]) is True
    assert _wait_for(lambda: collector.stats()["sampled"] == 1)
    collector.close()
    files = list((tmp_path / "draft-vocab" / "m1").glob("counts-*.json"))
    assert len(files) == 1


def test_restart_merges_previous_month_file(tmp_path: Path) -> None:
    first = _collector(tmp_path, sample_every=1)
    _with_tokenizer(first, "/t.json")
    first.observe("m1", "/t.json", ["alfa beta"])
    first.close()

    second = _collector(tmp_path, sample_every=1)
    _with_tokenizer(second, "/t.json")
    second.observe("m1", "/t.json", ["alfa beta"])
    second.close()

    files = list((tmp_path / "draft-vocab" / "m1").glob("counts-*.json"))
    counts = json.loads(files[0].read_text())["counts"]
    assert sum(counts.values()) == 4
