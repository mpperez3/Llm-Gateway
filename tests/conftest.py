"""Shared pytest fixtures for the llm-server test suite.

The live/E2E autotuning tests need a real GGUF artifact.  Rather than pinning a
large instruct model, they request the smallest valid GGUF on the Hub via
``tests/_real_test_model.py``.  Selecting any ``real_model``-marked test
auto-downloads that tiny artifact once per session and skips cleanly when the
network is unavailable or downloads are disabled.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Redirect the request log before any test module imports the package: the
# log path is resolved once at import time, and a test run must never append
# its fixtures to the live api-requests.log.
_TEST_STATE_DIR = Path(tempfile.mkdtemp(prefix="llm-server-tests-"))
os.environ.setdefault("LLM_SERVER_REQUESTS_LOG", str(_TEST_STATE_DIR / "api-requests.log"))
os.environ.setdefault("LLM_SERVER_LAST_CHAT_RESPONSE_LOG", str(_TEST_STATE_DIR / "last-chat-response.json"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _real_test_model import (  # noqa: E402
    ModelArtifactUnavailable,
    describe,
    ensure_real_test_model,
    model_path,
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_model: requires a real GGUF artifact (auto-downloaded tiny model)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    real_items = [item for item in items if item.get_closest_marker("real_model")]
    if not real_items:
        return
    try:
        ensure_real_test_model()
    except ModelArtifactUnavailable as exc:
        skip = pytest.mark.skip(reason=f"real test model unavailable: {exc}")
        for item in real_items:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def real_test_model() -> Path:
    """Session-scoped path to the tiny real GGUF, downloaded on demand."""
    try:
        return ensure_real_test_model()
    except ModelArtifactUnavailable as exc:
        pytest.skip(f"real test model unavailable: {exc} ({describe()})")


@pytest.fixture(autouse=True)
def _forbid_live_service_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test instead of letting it restart the live llm-server units.

    ``restart_service_to_free_vram`` and the installer helpers shell out to
    ``systemctl`` with the real unit names even when they are given a temporary
    layout. One missing mock in a test was enough to stop the live router four
    times and drop every connected client.
    """
    real_run = subprocess.run
    protected = ("llm-server-", "heimdall-gateway-")
    actions = {"restart", "stop", "start", "enable", "disable", "kill"}

    def guarded_run(cmd, *args, **kwargs):
        text = " ".join(str(part) for part in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        tokens = set(text.split())
        if "systemctl" in tokens and tokens & actions and any(name in text for name in protected):
            raise AssertionError(f"test attempted to control live systemd units: {text}")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)


@pytest.fixture(autouse=True)
def _isolate_uninstall_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep uninstall tests from deleting the live wrappers under the real home."""
    from llamacpp_stack import uninstall as _uninstall

    real_collect = _uninstall._collect_bin_targets
    temp_root = Path(tempfile.gettempdir()).resolve()

    def guarded(layout):
        allowed = [
            temp_root,
            Path(layout.bin_dir),
            Path(layout.install_root),
            Path(layout.state_dir),
            Path(layout.config_dir),
            Path(layout.run_dir),
        ]
        return [target for target in real_collect(layout) if any(_is_under(target, root) for root in allowed)]

    monkeypatch.setattr(_uninstall, "_collect_bin_targets", guarded)


def _is_under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


__all__ = ["real_test_model", "model_path"]
