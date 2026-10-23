#!/usr/bin/env python3
"""Provision the tiny real GGUF used by the live/E2E autotuning tests.

The old live tests pointed at ``models/SmolLM2-135M-Instruct-Q2_K.gguf``
(~88 MB) and, when a bigger model happened to be dropped in ``models/``,
the suite inherited its download/load cost.  The live benchmark and the
VRAM/repair integration tests only need *any* valid, loadable GGUF: they read
its on-disk size and (for the gated live run) hand it to ``llama-server``.

This module therefore defaults to the smallest valid GGUF published on the
Hugging Face Hub:

    ggml-org/models :: tinyllamas/stories260K.gguf   (~1.2 MB)

It is downloaded on demand into ``<repo>/models/`` (already git-ignored) and
reused across the whole test session.  Override the artifact with environment
variables when a richer model is genuinely required:

    LLM_SERVER_TEST_GGUF_REPO   (default ``ggml-org/models``)
    LLM_SERVER_TEST_GGUF_FILE   (default ``tinyllamas/stories260K.gguf``)
    LLM_SERVER_TEST_GGUF_NAME   (default ``stories260K.gguf``, flat name)
    LLM_SERVER_TEST_MODEL_DIR   (default ``<repo>/models``)
    LLM_SERVER_TEST_MODEL_OFFLINE=1   never hit the network; skip the tests

CLI: ``python tests/_real_test_model.py`` downloads (if needed) and prints the
absolute path, so CI can warm the artifact before running pytest.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPO = "ggml-org/models"
DEFAULT_FILE = "tinyllamas/stories260K.gguf"
DEFAULT_NAME = Path(DEFAULT_FILE).name

_OFFLINE_VALUES = {"1", "true", "yes", "on"}


class ModelArtifactUnavailable(RuntimeError):
    """Raised when the tiny real GGUF is missing and cannot be provisioned."""


def repo_id() -> str:
    return os.environ.get("LLM_SERVER_TEST_GGUF_REPO", DEFAULT_REPO).strip() or DEFAULT_REPO


def source_filename() -> str:
    return os.environ.get("LLM_SERVER_TEST_GGUF_FILE", DEFAULT_FILE).strip() or DEFAULT_FILE


def target_name() -> str:
    explicit = os.environ.get("LLM_SERVER_TEST_GGUF_NAME", "").strip()
    if explicit:
        return explicit
    return Path(source_filename()).name or DEFAULT_NAME


def model_dir() -> Path:
    explicit = os.environ.get("LLM_SERVER_TEST_MODEL_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return REPO_ROOT / "models"


def model_path() -> Path:
    """Return the path the tiny GGUF is expected at, without downloading."""
    return model_dir() / target_name()


def downloads_enabled() -> bool:
    return os.environ.get("LLM_SERVER_TEST_MODEL_OFFLINE", "").strip().lower() not in _OFFLINE_VALUES


def ensure_real_test_model(*, allow_download: bool | None = None) -> Path:
    """Return the tiny GGUF path, downloading it once if necessary.

    Raises :class:`ModelArtifactUnavailable` when the file is absent and downloads
    are disabled or fail, so callers can skip cleanly instead of erroring.
    """

    path = model_path()
    if path.exists() and path.stat().st_size > 0:
        return path

    if allow_download is None:
        allow_download = downloads_enabled()
    if not allow_download:
        raise ModelArtifactUnavailable(
            f"real test model missing and downloads disabled: {path} "
            "(unset LLM_SERVER_TEST_MODEL_OFFLINE or run tests/_real_test_model.py)"
        )

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ModelArtifactUnavailable(f"huggingface_hub is not importable: {exc}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cached = Path(hf_hub_download(repo_id=repo_id(), filename=source_filename()))
    except Exception as exc:  # pragma: no cover - network dependent
        raise ModelArtifactUnavailable(
            f"could not download {repo_id()}/{source_filename()}: {exc}"
        ) from exc

    if cached.resolve() != path.resolve():
        shutil.copy2(cached, path)

    if not path.exists() or path.stat().st_size == 0:
        raise ModelArtifactUnavailable(f"download produced no artifact at {path}")

    return path


def describe() -> str:
    return f"{repo_id()}/{source_filename()} -> {model_path()}"


def main() -> int:
    try:
        path = ensure_real_test_model()
    except ModelArtifactUnavailable as exc:
        print(f"test model unavailable: {exc}", file=sys.stderr)
        return 2
    size_mib = path.stat().st_size / (1024 * 1024)
    print(f"{path} ({size_mib:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
