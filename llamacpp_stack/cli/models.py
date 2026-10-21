"""Models/types extracted from llamacpp_stack/cli.py:1384-1543.

Spinner, LoadingBar, ManagedModel, ReplicaConfig, ReplicaRecord, ProbeTraceMetrics.
No top-level import of command_router/managed_commands to avoid cycles.
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from .constants import (
    DEFAULT_CTX_SIZE,
    DEFAULT_IDLE_TTL,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_TENSOR_SPLIT,
)


class Spinner:
    """Universal ASCII Spinner for maximum terminal compatibility."""

    def __init__(self, label="assistant: "):
        self.label = label
        self._frames = itertools.cycle(["|", "/", "-", "\\"])
        self._running = False
        self._thread = None
        self.cyan = "\033[36;1m"
        self.reset = "\033[0m"

    def _spin(self):
        sys.stdout.write("\033[?25l")
        while self._running:
            sys.stdout.write(f"\r{self.label}{self.cyan}{next(self._frames)}{self.reset}")
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write(f"\r{self.label}\033[K")
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        self._thread = None


class LoadingBar:
    """Indeterminate loading bar for model warmup."""

    def __init__(self, label="Loading model: ", width=24):
        self.label = label
        self.width = width
        self._running = False
        self._thread = None
        self.cyan = "\033[36;1m"
        self.reset = "\033[0m"

    def _render_frame(self, pos):
        cells = ["-"] * self.width
        for idx in range(4):
            cell = pos + idx
            if 0 <= cell < self.width:
                cells[cell] = "="
        return "[" + "".join(cells) + "]"

    def _spin(self):
        sys.stdout.write("\033[?25l")
        travel = max(1, self.width - 3)
        pos = 0
        direction = 1
        while self._running:
            bar = self._render_frame(pos)
            sys.stdout.write(f"\r{self.label}{self.cyan}{bar}{self.reset}")
            sys.stdout.flush()
            time.sleep(0.08)
            pos += direction
            if pos >= travel:
                pos = travel
                direction = -1
            elif pos <= 0:
                pos = 0
                direction = 1

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write(f"\r{self.label}\033[K")
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        self._thread = None


@dataclass
class ManagedModel:
    model_id: str
    repo_id: str
    quant: str | None
    filename: str
    local_path: str
    backend: str = "llama.cpp"
    mmproj_filename: str | None = None
    mmproj_path: str | None = None
    load_capabilities: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    ctx_size: int = DEFAULT_CTX_SIZE
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS
    tensor_split: str = DEFAULT_TENSOR_SPLIT
    host: str = "127.0.0.1"
    jinja: bool = True
    ttl: int = DEFAULT_IDLE_TTL
    description: str = ""
    downloaded_at: str = ""
    speculative: bool = False
    spec_variant_of: str | None = None
    spec_meta: dict[str, object] = field(default_factory=dict)
    auto_ctx_failed: bool = False
    auto_ctx_error: str = ""
    ctx_probe_read_s: float | None = None
    ctx_probe_tokens_s: float | None = None
    ctx_probe_totals_s: float | None = None
    ctx_probe_latency_ms: float | None = None
    ctx_probe_speed_tps: float | None = None
    ctx_probe_kv_gb: float | None = None
    ctx_probe_prompt_tokens: int | None = None
    server_overrides: dict[str, object] = field(default_factory=dict)


@dataclass
class ReplicaConfig:
    enabled: bool = False
    max: int = 1
    gpus_per_replica: int = 1
    placement: str = "exclusive_gpus"
    safety_vram_mib: int = 2048
    #: Capacity of one card, read from ``replicas.gpu_capacity_mib``. A configured
    #: figure rather than a probe, because the replica routes built from it are
    #: written into config.yaml and a live reading there would make two renders of
    #: one catalog disagree. 24576 MiB is a 24 GiB card: the common modern size, and
    #: an under-statement on anything larger, which only costs a replica.
    gpu_capacity_mib: int = 24576
    max_models_per_gpu: int = 2
    max_pack_fraction: float = 0.35
    sticky_ttl_s: int = 3600
    #: Policy knob, not a tuning knob. Serve a request with no live affinity on
    #: the base instead of spreading it to, or cold-starting, a second instance.
    #: Off by default: while another model is mid-load the same redirect already
    #: happens on its own, and that window is the only time it pays. Turning this
    #: on makes the redirect permanent, which costs genuine request parallelism
    #: whenever a base and a replica could both have served a conversation.
    prefer_base_over_replica: bool = False
    #: Deprecated. Kept so existing configs still parse. Request-path cold
    #: scale-out is now gated by real concurrency (a busy base), so a resident
    #: idle base always serves; use ``experimental.idle_gpu_warmer`` to keep a
    #: spare card warm proactively.
    idle_grace_s: int = 600
    #: Policy knob, reaper backstop. Retire a replica idle at least this long even
    #: when nothing else wants its GPU, so VRAM is handed back without waiting for
    #: llama-swap's own TTL. ``0`` disables the idle backstop entirely.
    max_idle_s: int = 1800


@dataclass
class ReplicaRecord:
    base_model_id: str
    replica_model_id: str
    gpu_set: list[int] = field(default_factory=list)
    status: str = "cold"
    estimated_mib: float | None = None
    actual_mib: float | None = None
    gpu_actual_mib: dict[int, float] = field(default_factory=dict)
    pid: int | None = None
    port: int | None = None
    in_flight: int = 0
    last_used: float = 0.0
    blacklist_until: float = 0.0
    #: Faults attributed to the *target*, not the request. Only
    #: FaultKind.TARGET_FATAL may increment it.
    fatal_faults: int = 0


@dataclass
class ProbeTraceMetrics:
    model_buffers_mib: dict[int, float] = field(default_factory=dict)
    kv_buffers_mib: dict[int, float] = field(default_factory=dict)
    compute_buffers_mib: dict[int, float] = field(default_factory=dict)
    projector_gpu: int | None = None
    oom_gpu: int | None = None
    oom_requested_mib: float | None = None


def model_weights_size_mib(model: ManagedModel) -> float | None:
    """Bytes ``model.local_path`` occupies on disk, as a whole tree or one file.

    ``None`` when the path cannot be measured at all; a caller that treats an
    unknown size as zero has to say so itself, because the two answers mean
    different things to a capacity check.
    """
    try:
        path = Path(model.local_path)
        if path.is_dir():
            total = sum(
                file_path.stat().st_size
                for file_path in path.rglob("*")
                if file_path.is_file()
            )
        else:
            total = path.stat().st_size
    except Exception:
        return None
    return total / (1024 * 1024)


def estimate_model_runtime_mib(model: ManagedModel) -> float | None:
    """Whole-process footprint of one loaded copy: weights, KV cache, buffers, overhead.

    This is the only figure that answers "will this fit", so every capacity decision
    compares against it; raw weights alone understate a process by the KV cache and
    compute buffers it needs. ``None`` when the weights cannot be measured, and an
    unmeasured model must never be admitted: an unknown cost is not a small cost.

    A pure function of the catalog - only ``local_path``, ``ctx_size`` and
    ``server_overrides`` - so two renders of one catalog agree.
    """
    size_mib = model_weights_size_mib(model)
    if size_mib is None:
        return None
    overrides = model.server_overrides or {}
    # server_overrides is dict[str, object]; the int() is kept so a string-valued
    # override such as "32768" still converts, and cast so the arithmetic below is typed.
    ctx = int(cast("int", overrides.get("ctx_size", model.ctx_size)) or model.ctx_size or 8192)
    parallel = int(cast("int", overrides.get("parallel", 1)) or 1)
    batch = int(cast("int", overrides.get("batch_size", 2048)) or 2048)
    ubatch = int(cast("int", overrides.get("ubatch_size", 512)) or 512)
    kv_mib = (ctx * parallel * (size_mib / 2048.0) * 0.5) / 1024.0
    compute_mib = (ubatch * 0.5) + (batch * 0.1)
    return (size_mib * 1.08) + kv_mib + compute_mib + 700.0


__all__ = [
    "Spinner",
    "LoadingBar",
    "ManagedModel",
    "ReplicaConfig",
    "ReplicaRecord",
    "ProbeTraceMetrics",
    "model_weights_size_mib",
    "estimate_model_runtime_mib",
]
