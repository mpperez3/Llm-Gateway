"""Runtime GPU placement: the planner, its hysteresis, and the safe config writer.

The GPU a route launches on is baked into its command by ``assign_model_gpu_sets``,
so "put this model on the free card" is not a routing decision llama-swap can make:
either the route's command is rewritten or the current occupant is retired. These
tests pin that decision down without touching a GPU.
"""

import argparse
import contextlib
import inspect
import json
import re
import sys
import tempfile
import textwrap
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import yaml

from llamacpp_stack import _cli_impl
from llamacpp_stack import cli as cli
from llamacpp_stack.cli import daemon as daemon_mod
from llamacpp_stack.cli import models as models_mod
from llamacpp_stack.cli import replica as R
from llamacpp_stack.cli.models import ReplicaConfig, ReplicaRecord

TARGET_ID = "wants-gpu-1"
OCCUPANT_ID = "other-model"
REPLICA_ID = f"{TARGET_ID}__replica_0"
SERVER = Path("/usr/bin/llama-server")

OCCUPANT_PID = 4242
OCCUPANT_MIB = 18218.0
NEEDED_MIB = 20045.0
ALT_FREE_MIB = 24069.0
SMALL_FREE_MIB = 9000.0

# The switch's VRAM gate is model_has_enough_free_vram_to_load, which adds a 1024
# MiB per-GPU overhead and the 2048 MiB safety cushion to the estimate.
ROUTE_VARIANT_NEEDED_MIB = 4000.0
CARD_TOTAL_MIB = 32000.0
CARD_FITS_FREE_MIB = 24000.0
CARD_BUSY_FREE_MIB = 3000.0


def _monotonic() -> float:
    """Overrides are stamped with time.monotonic, so tests must start from the real
    clock or every live assertion reads as already expired."""
    return _cli_impl.time.monotonic()


def _make_model(model_id: str, local_path: str, tensor_split: str = "1"):
    return R.ManagedModel(
        model_id=model_id,
        repo_id="org/model",
        quant="EXL3_3.00bpw",
        filename="model.safetensors",
        local_path=local_path,
        backend="llamacpp",
        mmproj_filename=None,
        mmproj_path=None,
        load_capabilities=[],
        aliases=[],
        ctx_size=262144,
        # build_llama_server_command int()s this, so None would raise.
        n_gpu_layers=-1,
        tensor_split=tensor_split,
        host=None,
        jinja=True,
        ttl=None,
        description=None,
        downloaded_at=None,
        speculative=False,
        spec_variant_of=None,
        spec_meta=None,
        auto_ctx_failed=False,
        auto_ctx_error=None,
    )


class _RouterStateMixin:
    """The router state is a process-lifetime singleton; keep tests independent."""

    def _clear_placement_state(self):
        state = _cli_impl.REPLICA_ROUTER_STATE
        with state.lock:
            state.gpu_override.clear()
            state.gpu_override_written_at.clear()
            state.gpu_demand.clear()
            state.base_in_flight.clear()
            state.records.clear()
        _cli_impl._PLACEMENT_WRITE_LOG.clear()

    def reset_placement_state(self):
        # The cleanup must be the non-registering variant: a self-registering one
        # re-arms itself on every cleanup and never finishes.
        self._clear_placement_state()
        self.addCleanup(self._clear_placement_state)


class PlannerTest(_RouterStateMixin, unittest.TestCase):
    def setUp(self):
        self.reset_placement_state()
        self.catalog = [_make_model(TARGET_ID, "/models/target.safetensors")]
        self.state = _cli_impl.REPLICA_ROUTER_STATE

    @contextlib.contextmanager
    def _planning(
        self,
        *,
        enabled: bool = True,
        repin: bool = False,
        loaded: bool = False,
        replica_sets=(),
        pinned=(1,),
        occupants=None,
        in_flight: int = 1,
        alt=(0,),
        alt_free: float = ALT_FREE_MIB,
        fits_alt: bool = False,
        claim: bool = True,
    ):
        if occupants is None:
            occupants = {pinned[0]: [(OCCUPANT_ID, OCCUPANT_PID, OCCUPANT_MIB)]} if pinned else {}
        stack = contextlib.ExitStack()
        patches = {
            "_runtime_placement_enabled": enabled,
            "_placement_repin_enabled": repin,
            "get_catalog_model_process": mock.Mock(
                return_value={"pid": 1} if loaded else None
            ),
            "get_model_replica_config": mock.Mock(
                return_value=ReplicaConfig(enabled=bool(replica_sets), max=2)
            ),
            "resolve_global_replica_config": mock.Mock(return_value={}),
            "_replica_gpu_sets": mock.Mock(return_value=list(replica_sets)),
            "effective_model_gpu_set": mock.Mock(return_value=list(pinned)),
            "_gpu_occupants": mock.Mock(return_value=dict(occupants)),
            "estimate_model_runtime_mib": mock.Mock(return_value=NEEDED_MIB),
            "instance_in_flight": mock.Mock(return_value=in_flight),
            "best_alternative_gpu": mock.Mock(return_value=(list(alt) if alt else None, alt_free)),
            "model_has_enough_free_vram_to_load": mock.Mock(return_value=(fits_alt, {"reason": "stub"})),
            "_claim_placement_write": mock.Mock(return_value=claim),
        }
        with stack:
            for name, value in patches.items():
                if callable(value):
                    stack.enter_context(mock.patch.object(_cli_impl, name, value))
                else:
                    stack.enter_context(
                        mock.patch.object(_cli_impl, name, mock.Mock(return_value=value))
                    )
            yield stack

    def _plan(self):
        return _cli_impl.plan_runtime_placement(TARGET_ID, self.catalog)

    def test_a_busy_occupant_is_rejected_and_the_message_names_its_card(self):
        with self._planning(in_flight=2, alt=None, alt_free=0.0):
            plan = self._plan()
        self.assertEqual(plan.action, "reject")
        self.assertEqual(plan.pinned, [1])
        self.assertEqual(plan.occupant, (OCCUPANT_ID, OCCUPANT_PID, int(OCCUPANT_MIB)))
        self.assertIn("pinned GPU 1", plan.message)
        self.assertIn(OCCUPANT_ID, plan.message)
        self.assertIn(str(OCCUPANT_PID), plan.message)
        self.assertIn("2 requests in flight", plan.message)

    def test_an_already_loaded_model_is_never_replanned(self):
        with self._planning(loaded=True, repin=True, fits_alt=True, in_flight=0):
            plan = self._plan()
        self.assertEqual(plan.action, "forward")

    def test_an_idle_occupant_with_nowhere_else_to_go_is_retired(self):
        with self._planning(in_flight=0, alt=None, alt_free=0.0):
            plan = self._plan()
        self.assertEqual(plan.action, "retire_then_forward")
        self.assertIn(f"Unloaded idle '{OCCUPANT_ID}'", plan.message)

    def test_a_free_card_that_fits_is_used_instead_of_unloading_the_occupant(self):
        with self._planning(in_flight=0, repin=True, fits_alt=True):
            plan = self._plan()
        self.assertEqual(plan.action, "repin_then_forward")
        self.assertEqual(plan.alt_gpu_set, [0])
        self.assertIn("Moved the model to GPU 0", plan.message)
        self.assertIn(f"instead of unloading '{OCCUPANT_ID}'", plan.message)

    def test_repinning_is_refused_unless_the_kill_switch_allows_it(self):
        with self._planning(in_flight=0, repin=False, fits_alt=True):
            plan = self._plan()
        self.assertEqual(plan.action, "retire_then_forward")

    def test_replica_mode_never_repins_a_base(self):
        with self._planning(replica_sets=[[0]], in_flight=0, repin=True, fits_alt=True):
            plan = self._plan()
        self.assertEqual(plan.action, "forward")

    def test_the_replica_guard_is_given_the_static_assignment(self):
        seen = {}

        def spy(model, cfg, total_gpus=None, *, base_gpu_set=None, reserved_gpu_set=None):
            seen["base_gpu_set"] = base_gpu_set
            seen["reserved_gpu_set"] = reserved_gpu_set
            return []

        with self._planning(replica_sets=[[0]], in_flight=0):
            with mock.patch.object(_cli_impl, "_replica_gpu_sets", spy), mock.patch.object(
                _cli_impl,
                "cached_model_gpu_sets",
                mock.Mock(return_value={TARGET_ID: [1], OCCUPANT_ID: [1]}),
            ), mock.patch.object(_cli_impl, "reserved_base_gpus", mock.Mock(return_value={0, 1})):
                plan = self._plan()
        self.assertEqual(seen["base_gpu_set"], [1])
        self.assertEqual(seen["reserved_gpu_set"], {0, 1})
        self.assertNotEqual(plan.action, "forward")

    def test_the_planner_is_off_unless_the_kill_switch_enables_it(self):
        with self._planning(enabled=False, in_flight=0, repin=True, fits_alt=True):
            plan = self._plan()
        self.assertEqual(plan.action, "forward")

    def test_the_shortfall_is_named_when_no_card_is_big_enough(self):
        with self._planning(in_flight=1, alt=(0,), alt_free=SMALL_FREE_MIB, fits_alt=False):
            plan = self._plan()
        self.assertEqual(plan.action, "reject")
        self.assertEqual(plan.alt_gpu_set, [0])
        self.assertEqual(plan.alt_free_mib, SMALL_FREE_MIB)
        expected = int(NEEDED_MIB + 1024 + 2048 - SMALL_FREE_MIB)
        self.assertIn(f"short by {expected} MiB", plan.message)
        self.assertIn("GPU 0 has 9000 MiB free", plan.message)

    def test_an_idle_pin_is_left_alone(self):
        with self._planning(occupants={}):
            plan = self._plan()
        self.assertEqual(plan.action, "forward")
        self.assertEqual(plan.pinned, [1])

    def test_the_capacity_check_sees_the_pinned_card_not_the_launch_default(self):
        """model_launch_gpu_set answers [0] for every single-GPU model; trusting it
        measures the wrong card and refuses a model whose own card is empty."""
        seen = []

        def _spy(model, **kwargs):
            seen.append(kwargs.get("gpu_set"))
            return True, {"reason": "stub"}

        with self._planning(in_flight=0, repin=True, alt=(0,)) as stack:
            stack.enter_context(
                mock.patch.object(
                    _cli_impl, "model_has_enough_free_vram_to_load", side_effect=_spy
                )
            )
            stack.enter_context(
                mock.patch.object(
                    _cli_impl,
                    "model_launch_gpu_set",
                    side_effect=AssertionError("launch default must not decide placement"),
                )
            )
            plan = self._plan()
        self.assertEqual(plan.action, "repin_then_forward")
        self.assertEqual(seen, [[0]], "the alternative card is the one that gets measured")


class HysteresisTest(_RouterStateMixin, unittest.TestCase):
    def setUp(self):
        self.reset_placement_state()
        self.state = _cli_impl.REPLICA_ROUTER_STATE
        self.catalog = [_make_model(TARGET_ID, "/models/target.safetensors")]

    def test_a_live_override_outranks_the_static_map(self):
        with mock.patch.object(
            _cli_impl, "cached_model_gpu_sets", return_value={TARGET_ID: [1]}
        ):
            self.assertEqual(_cli_impl.effective_model_gpu_set(TARGET_ID, self.catalog), [1])
            self.assertTrue(self.state.claim_gpu_override(TARGET_ID, [0]))
            self.assertEqual(_cli_impl.effective_model_gpu_set(TARGET_ID, self.catalog), [0])

    def test_an_expired_override_falls_back_to_the_static_map(self):
        now = _monotonic()
        with mock.patch.object(
            _cli_impl, "cached_model_gpu_sets", return_value={TARGET_ID: [1]}
        ):
            self.state.claim_gpu_override(TARGET_ID, [0], ttl_s=10.0, now=now)
            self.assertEqual(_cli_impl.effective_model_gpu_set(TARGET_ID, self.catalog), [0])
            self.assertIsNone(self.state.live_gpu_override(TARGET_ID, now=now + 11.0))
            with mock.patch.object(_cli_impl.time, "monotonic", return_value=now + 11.0):
                self.assertEqual(_cli_impl.effective_model_gpu_set(TARGET_ID, self.catalog), [1])

    def test_prune_drops_an_expired_override(self):
        now = _monotonic()
        self.state.claim_gpu_override(TARGET_ID, [0], ttl_s=10.0, now=now)
        self.state.prune(now=now + 11.0)
        self.assertEqual(self.state.gpu_override, {})
        self.assertEqual(self.state.gpu_override_written_at, {})

    def test_a_second_override_while_one_is_live_is_refused(self):
        now = _monotonic()
        self.assertTrue(self.state.claim_gpu_override(TARGET_ID, [0], now=now))
        self.assertFalse(self.state.claim_gpu_override(TARGET_ID, [1], now=now + 1.0))

    def test_a_second_write_inside_the_minimum_interval_is_refused(self):
        now = _monotonic()
        self.assertTrue(self.state.claim_gpu_override(TARGET_ID, [0], ttl_s=1.0, now=now))
        self.assertFalse(
            self.state.claim_gpu_override(TARGET_ID, [1], now=now + 10.0),
            "the TTL is already spent, so only the minimum interval can refuse this",
        )

    def test_the_hourly_budget_caps_config_rewrites(self):
        clock = _monotonic()
        models = [f"{TARGET_ID}-{index}" for index in range(_cli_impl.PLACEMENT_MAX_WRITES_PER_HOUR + 1)]
        with mock.patch.object(_cli_impl.time, "monotonic", side_effect=lambda: clock):
            for model_id in models[:-1]:
                self.assertTrue(_cli_impl._claim_placement_write(model_id, [0]), model_id)
            self.assertFalse(
                _cli_impl._claim_placement_write(models[-1], [0]),
                "the budget is global, not per model",
            )
            clock += 3601.0
            self.assertTrue(
                _cli_impl._claim_placement_write(models[-1], [0]),
                "entries older than an hour must stop counting against the budget",
            )


class SafeWriterTest(_RouterStateMixin, unittest.TestCase):
    def setUp(self):
        self.reset_placement_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "config.yaml"
        # Above the small-model threshold, so the matrix gives it its own group.
        self.big = self.root / "big.safetensors"
        with open(self.big, "wb") as handle:
            handle.truncate(6 * 1024 * 1024 * 1024)
        self.model = _make_model(TARGET_ID, str(self.big))
        self.other = _make_model(OCCUPANT_ID, str(self.root / "other.safetensors"))
        self.catalog = [self.model, self.other]
        R.render_llamaswap_config(self.catalog, self.config_path, SERVER, 11436, 300, {}, {})

    def _ensure(self, gpu_set=None):
        R.ensure_internal_route_in_llamaswap_config(
            self.model,
            TARGET_ID,
            self.catalog,
            self.config_path,
            SERVER,
            300,
            {},
            gpu_set=gpu_set,
        )

    def _read(self):
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def test_a_corrupt_config_is_never_overwritten(self):
        """A failed read used to become ``{}`` and publish a one-route catalog that
        the manager cannot repair, because the full render only runs at startup."""
        cases = {
            "truncated_yaml": "models:\n  a: [1, 2\n",
            "a_yaml_list": "- one\n- two\n",
            "an_empty_file": "",
            "models_not_a_mapping": "models:\n  - one\n",
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                self.config_path.write_text(body, encoding="utf-8")
                before = self.config_path.read_bytes()
                with self.assertRaises(R.LlamaSwapConfigReadError):
                    R._read_config_strict(self.config_path)
                with self.assertRaises(R.LlamaSwapConfigReadError):
                    self._ensure()
                self.assertEqual(self.config_path.read_bytes(), before)

    def test_a_missing_config_reads_as_empty_rather_than_raising(self):
        self.assertEqual(R._read_config_strict(self.root / "absent.yaml"), {})

    def test_a_gpu_only_change_reaches_the_file(self):
        """The old early-exit compared the projector state alone, so a CVD-only
        rewrite of a route was silently dropped."""
        with mock.patch.object(
            R, "cached_model_gpu_sets", return_value={TARGET_ID: [1], OCCUPANT_ID: [1]}
        ):
            R.render_llamaswap_config(self.catalog, self.config_path, SERVER, 11436, 300, {}, {})
            self.assertIn("CUDA_VISIBLE_DEVICES=1", self._read()["models"][TARGET_ID]["cmd"])
            self._ensure(gpu_set=[0])
            after = self._read()["models"][TARGET_ID]["cmd"]
        self.assertIn("CUDA_VISIBLE_DEVICES=0", after)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=1", after)

    def test_a_base_route_can_be_moved_to_another_card(self):
        with mock.patch.object(R, "cached_model_gpu_sets", return_value={TARGET_ID: [1]}):
            self._ensure(gpu_set=[0])
        cmd = self._read()["models"][TARGET_ID]["cmd"]
        self.assertIn("CUDA_VISIBLE_DEVICES=0", cmd)

    def test_the_moved_pair_becomes_co_loadable(self):
        with mock.patch.object(
            R,
            "cached_model_gpu_sets",
            return_value={TARGET_ID: [1], OCCUPANT_ID: [1]},
        ):
            self._ensure(gpu_set=[0])
        membership = _cli_impl._matrix_group_membership(self.config_path)
        self.assertTrue(
            set(membership.get(TARGET_ID) or ()) & set(membership.get(OCCUPANT_ID) or ()),
            f"a model sharing a set name is co-loadable; got {membership}",
        )

    def test_concurrent_writers_never_shrink_or_corrupt_the_model_set(self):
        errors = []
        barrier = threading.Barrier(2)

        def _hammer(fn):
            try:
                barrier.wait()
                for _ in range(50):
                    fn()
            except Exception as exc:  # noqa: BLE001 - surfaced by the assert below
                errors.append(exc)

        def _render():
            R.render_llamaswap_config(
                self.catalog, self.config_path, SERVER, 11436, 300, {}, {}
            )

        threads = [
            threading.Thread(target=_hammer, args=(_render,)),
            threading.Thread(target=_hammer, args=(lambda: self._ensure(gpu_set=[0]),)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        data = self._read()
        published = set(data["models"])
        self.assertTrue(set(self.catalog_ids()) <= published)
        self.assertEqual(
            sorted(name for name in published if name not in set(self.catalog_ids())),
            ["wants-gpu-1__gpu0", "wants-gpu-1__gpu1"],
        )
        self.assertEqual(list(self.root.glob("config.yaml.*.tmp")), [])
        self.assertFalse((self.root / "config.tmp").exists())

    def catalog_ids(self):
        return {model.model_id for model in self.catalog}


class CompletionsAdmissionTest(unittest.TestCase):
    def test_the_completions_route_has_the_same_admission_gate(self):
        """/v1/completions resolves, picks a replica and forwards without ever
        calling _reject_if_gpu_busy, so it can trigger the very eviction the
        planner exists to avoid."""
        source = inspect.getsource(_cli_impl.start_ctx_metadata_server)
        start = source.index("def _proxy_request(self,")
        # It is the last method of the nested Handler, so the body runs to the end.
        body = textwrap.dedent(source[start:])
        self.assertIn("_reject_if_gpu_busy", body)


class PlacementConfigTest(_RouterStateMixin, unittest.TestCase):
    def setUp(self):
        self.reset_placement_state()
        self.args = argparse.Namespace(config="/dev/null")

    def _with_experimental(self, payload):
        return mock.patch.object(
            _cli_impl, "_load_server_config_payload", return_value=payload
        )

    def test_it_defaults_to_off(self):
        with self._with_experimental({}):
            self.assertEqual(
                _cli_impl._resolve_runtime_placement_config(self.args),
                {"enabled": False, "mode": "off"},
            )

    def test_disabling_forces_the_mode_back_to_off(self):
        with self._with_experimental(
            {"experimental": {"runtime_placement": {"enabled": False, "mode": "repin"}}}
        ):
            self.assertFalse(_cli_impl._placement_repin_enabled(self.args))

    def test_an_unknown_mode_is_treated_as_off(self):
        with self._with_experimental(
            {"experimental": {"runtime_placement": {"enabled": True, "mode": "yolo"}}}
        ):
            self.assertFalse(_cli_impl._runtime_placement_enabled(self.args))

    def test_repin_is_only_on_in_repin_mode(self):
        payload = {"experimental": {"runtime_placement": {"enabled": True, "mode": "retire_only"}}}
        with self._with_experimental(payload):
            self.assertTrue(_cli_impl._runtime_placement_enabled(self.args))
            self.assertFalse(_cli_impl._placement_repin_enabled(self.args))
        payload["experimental"]["runtime_placement"]["mode"] = "repin"
        with self._with_experimental(payload):
            self.assertTrue(_cli_impl._placement_repin_enabled(self.args))

    def test_a_missing_server_config_leaves_it_off(self):
        with mock.patch.object(
            _cli_impl, "_load_server_config_payload", side_effect=RuntimeError("boom")
        ):
            self.assertFalse(_cli_impl._runtime_placement_enabled(self.args))


class ConflictScopeTest(unittest.TestCase):
    """The conflict list must only cover the cards the launch will actually use.

    Regression: a base model pinned to GPU 1 was refused with "the GPU is already
    in use: <model on GPU 0>" while its own card sat empty, so two models could
    never be resident at once even with a whole card free.
    """

    @staticmethod
    def _conflict_source() -> str:
        source = inspect.getsource(_cli_impl.get_gpu_conflict_message)
        body = source[source.index("gpu_process_map = get_gpu_process_map()") :]
        return body[: body.index("if not conflicts:")]

    def test_the_per_gpu_map_is_not_replica_only(self):
        self.assertNotIn(
            "if replica_gpu_set is not None:",
            self._conflict_source(),
            "the per-GPU map was gated on replica_gpu_set, so base models saw no GPU "
            "per pid and every live process became a conflict",
        )

    def test_conflicts_are_scoped_to_the_effective_gpu_set(self):
        body = self._conflict_source()
        self.assertIn("conflict_scope = target_gpu_set", body)
        self.assertNotIn(
            "gpus.isdisjoint(set(replica_gpu_set))",
            body,
            "scoping by replica_gpu_set leaves base models unscoped",
        )


def _big_target_model(root: Path):
    """A model above the small-model threshold, or gpu_route_variants publishes no
    per-card route for it and there is nothing to switch to."""
    weights = root / "target.safetensors"
    with open(weights, "wb") as handle:
        handle.truncate(6 * 1024 * 1024 * 1024)
    return _make_model(TARGET_ID, str(weights))


@contextlib.contextmanager
def _fake_gateway(*, catalog, occupant_cards=(), published=(1,), free=None, loaded=None, loaded_error=None):
    """Fake every probe the switch reads, so nothing here needs a GPU.

    occupant_cards pairs a card with the model sitting on it, published lists
    the __gpuN cards llama-swap answers for, free is free MiB per card, and
    loaded lists the route ids llama-swap reports as running - by default the
    routes of the models in occupant_cards, since one process per model runs
    them. Yields the mocks so a test can read what was logged.
    """
    published_ids = {model.model_id for model in catalog}
    published_ids.update(f"{TARGET_ID}__gpu{gpu}" for gpu in published)
    processes = []
    per_pid: dict[int, dict[int, float]] = {}
    for index, (gpu, occupant) in enumerate(occupant_cards):
        pid = OCCUPANT_PID + index
        processes.append(
            {
                "pid": pid,
                "cmdline": f"llama-server --model {occupant.local_path} --port {12000 + pid}",
                "model_path": str(Path(occupant.local_path).resolve()),
                "port": 12000 + pid,
            }
        )
        per_pid[pid] = {gpu: OCCUPANT_MIB}
    if loaded is None:
        loaded = {occupant.model_id for _gpu, occupant in occupant_cards}
    free_mib = {0: CARD_FITS_FREE_MIB, 1: CARD_FITS_FREE_MIB} if free is None else dict(free)
    patches = {
        "detect_cuda_device_count": mock.Mock(return_value=2),
        "cached_model_gpu_sets": mock.Mock(
            return_value={model.model_id: [0] for model in catalog}
        ),
        "get_published_model_ids": mock.Mock(return_value=published_ids),
        "get_llama_server_processes": mock.Mock(return_value=processes),
        "get_loaded_model_ids": mock.Mock(return_value=set(loaded), side_effect=loaded_error),
        "get_gpu_process_memory_by_pid": mock.Mock(return_value=per_pid),
        "_is_ollama_process": mock.Mock(return_value=False),
        "estimate_model_runtime_mib": mock.Mock(return_value=ROUTE_VARIANT_NEEDED_MIB),
        "_query_gpu_memory_snapshot_cached": mock.Mock(
            return_value={
                gpu: {
                    "used_mib": 0.0,
                    "free_mib": float(free_mib.get(gpu, 0.0)),
                    "total_mib": CARD_TOTAL_MIB,
                }
                for gpu in (0, 1)
            }
        ),
        "log_api_event": mock.Mock(),
    }
    with contextlib.ExitStack() as stack:
        for name, value in patches.items():
            stack.enter_context(mock.patch.object(_cli_impl, name, value))
        yield patches


class RouteSwitchTest(_RouterStateMixin, unittest.TestCase):
    """The routing switch: which already-published route a request must load."""

    def setUp(self):
        self.reset_placement_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = _big_target_model(self.root)
        self.variant_id = f"{TARGET_ID}__gpu1"

    def _occupant(self, model_id: str):
        return _make_model(model_id, str(self.root / f"{model_id}.safetensors"))

    _switch = staticmethod(_fake_gateway)

    def _select(self, catalog, *, allow_variant: bool = True):
        return _cli_impl.select_model_route(
            TARGET_ID,
            catalog,
            Path("/dev/null"),
            "127.0.0.1",
            11436,
            allow_variant=allow_variant,
        )

    def _logged_kinds(self, log_mock) -> list[str]:
        return [call.args[0] for call in log_mock.call_args_list]

    def test_a_free_pinned_card_keeps_the_base_route(self):
        with self._switch(catalog=[self.target]):
            self.assertEqual(self._select([self.target]), (TARGET_ID, "pinned_free"))

    def test_a_busy_pin_moves_to_the_published_route_of_a_free_card(self):
        occupant = self._occupant(OCCUPANT_ID)
        with self._switch(
            catalog=[self.target, occupant],
            occupant_cards=[(0, occupant)],
            free={0: CARD_BUSY_FREE_MIB, 1: CARD_FITS_FREE_MIB},
        ):
            route_id, reason = self._select([self.target, occupant])
        self.assertEqual(route_id, self.variant_id)
        self.assertEqual(reason, "variant_gpu1")

    def test_a_card_without_a_published_route_is_never_invented(self):
        occupant = self._occupant(OCCUPANT_ID)
        with self._switch(
            catalog=[self.target, occupant],
            occupant_cards=[(0, occupant)],
            published=(),
            free={0: CARD_BUSY_FREE_MIB, 1: CARD_FITS_FREE_MIB},
        ):
            self.assertEqual(self._select([self.target, occupant]), (TARGET_ID, "static"))

    def test_a_occupied_variant_card_is_skipped_even_when_it_has_room(self):
        on_pin = self._occupant(OCCUPANT_ID)
        on_variant = self._occupant("third-model")
        catalog = [self.target, on_pin, on_variant]
        with self._switch(
            catalog=catalog,
            occupant_cards=[(0, on_pin), (1, on_variant)],
            free={0: CARD_BUSY_FREE_MIB, 1: CARD_FITS_FREE_MIB},
        ):
            route_id, reason = self._select(catalog)
        self.assertEqual(route_id, TARGET_ID)
        self.assertEqual(reason, "pinned_busy")

    def test_an_already_loaded_model_is_never_moved(self):
        occupant = self._occupant(OCCUPANT_ID)
        catalog = [self.target, occupant]
        with self._switch(
            catalog=catalog,
            occupant_cards=[(0, self.target)],
            free={0: CARD_BUSY_FREE_MIB, 1: CARD_FITS_FREE_MIB},
        ):
            route_id, reason = self._select(catalog)
        self.assertEqual(route_id, TARGET_ID)
        self.assertEqual(reason, "already_loaded")

    def test_the_live_copy_of_an_already_loaded_model_is_the_one_that_answers(self):
        # The target is running on the variant card while its pinned card is free:
        # returning the base id would make llama-swap load a second copy there.
        catalog = [self.target]
        with self._switch(
            catalog=catalog, occupant_cards=[(1, self.target)], loaded={self.variant_id}
        ):
            self.assertEqual(
                self._select(catalog), (self.variant_id, "already_loaded_variant")
            )

    def test_the_lowest_card_wins_when_two_variants_of_one_model_are_live(self):
        catalog = [self.target]
        with self._switch(
            catalog=catalog,
            occupant_cards=[(1, self.target)],
            published=(1, 2),
            loaded={f"{TARGET_ID}__gpu2", self.variant_id},
        ) as probes:
            self.assertEqual(
                self._select(catalog), (self.variant_id, "already_loaded_variant")
            )
        self.assertIn("route_multiple_live_routes", self._logged_kinds(probes["log_api_event"]))

    def test_a_live_variant_the_switch_never_published_is_not_forwarded_to(self):
        catalog = [self.target]
        with self._switch(
            catalog=catalog,
            occupant_cards=[(1, self.target)],
            loaded={f"{TARGET_ID}__gpu2"},
        ):
            self.assertEqual(self._select(catalog), (TARGET_ID, "already_loaded"))

    def test_two_live_routes_of_one_model_need_the_base_and_an_anomaly_event(self):
        catalog = [self.target]
        with self._switch(
            catalog=catalog,
            occupant_cards=[(1, self.target)],
            loaded={TARGET_ID, self.variant_id},
        ) as probes:
            self.assertEqual(self._select(catalog), (TARGET_ID, "already_loaded"))
        self.assertIn("route_multiple_live_routes", self._logged_kinds(probes["log_api_event"]))

    def test_a_live_route_lookup_that_fails_keeps_the_base_route(self):
        catalog = [self.target]
        with self._switch(
            catalog=catalog,
            occupant_cards=[(1, self.target)],
            loaded={self.variant_id},
            loaded_error=RuntimeError("swap unreachable"),
        ) as probes:
            self.assertEqual(self._select(catalog), (TARGET_ID, "already_loaded"))
        self.assertIn("route_live_lookup_failed", self._logged_kinds(probes["log_api_event"]))

    def test_an_unknown_or_replica_model_never_switches(self):
        with self._switch(catalog=[self.target]):
            self.assertEqual(
                _cli_impl.select_model_route(
                    "not-in-catalog", [self.target], None, "127.0.0.1", 11436, allow_variant=True
                ),
                ("not-in-catalog", "static"),
            )
            self.assertEqual(
                _cli_impl.select_model_route(
                    REPLICA_ID, [self.target], None, "127.0.0.1", 11436, allow_variant=True
                ),
                (REPLICA_ID, "static"),
            )

    def test_the_switch_is_off_unless_it_is_allowed(self):
        occupant = self._occupant(OCCUPANT_ID)
        catalog = [self.target, occupant]
        with self._switch(
            catalog=catalog,
            occupant_cards=[(0, occupant)],
            free={0: CARD_BUSY_FREE_MIB, 1: CARD_FITS_FREE_MIB},
        ):
            self.assertEqual(self._select(catalog, allow_variant=False), (TARGET_ID, "static"))


class AlreadyLoadedVariantWiringTest(unittest.TestCase):
    """The shipped payload-rewrite block, run against a live variant.

    ``_proxy_request`` is nested inside ``start_ctx_metadata_server``'s closure, so
    the rewrite-and-book statements are lifted out of the real module and exec'd
    against the real resolver. A resolver that returns a __gpuN id is therefore the
    only thing that can put a variant id into the forwarded payload, and the booking
    below it still has to name the catalog model_id.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = _big_target_model(self.root)
        self.variant_id = f"{TARGET_ID}__gpu1"
        self.state = mock.Mock()

    @staticmethod
    def _block(source: str, first: str, last: str) -> str:
        """The handler's own lines from ``first`` up to (not including) ``last``,
        dedented to module level.

        Each slice is dedented on its own because the resolver call and the
        booking below it sit at different nesting levels in the handler, and
        dedent() is a no-op unless every line of the slice shares one prefix.
        """
        start = source.rindex("\n", 0, source.index(first)) + 1
        end = source.rindex("\n", 0, source.index(last, start)) + 1
        return textwrap.dedent(source[start:end])

    def _rewrite_and_book(self, catalog):
        source = inspect.getsource(_cli_impl.start_ctx_metadata_server)
        booked = "REPLICA_ROUTER_STATE.request_started(upstream_model_name)"
        gate = 'if (\n'
        block = "".join(
            (
                self._block(source, "route_on_variant = False", "if route_id != upstream_model_name:"),
                self._block(source, "if route_id != upstream_model_name:", gate),
                self._block(source, gate, booked),
                f"{booked}\n",
            )
        )
        payload = {"model": TARGET_ID, "messages": []}
        self.handler = mock.Mock()
        self.handler._reject_if_gpu_busy.return_value = False
        scope = {
            "select_model_route": _cli_impl.select_model_route,
            "upstream_model_name": TARGET_ID,
            "catalog": catalog,
            "args": argparse.Namespace(config=Path("/dev/null"), public_port=11436),
            "client_host": "127.0.0.1",
            "proxy_payload": payload,
            "body": b"",
            "json": json,
            "REPLICA_ROUTER_STATE": self.state,
            "self": self.handler,
            "is_replica_request": False,
            "model_entry": object(),
            "activity_model": TARGET_ID,
        }
        # Wrapped in a function because the gate's rejection path is a `return`,
        # which is not legal at module level.
        exec(compile("def _probe():\n" + textwrap.indent(block, "    "), "<_proxy_request>", "exec"), scope)
        scope["_probe"]()
        return payload

    def test_a_live_variant_is_rewritten_into_the_payload_and_the_base_is_booked(self):
        catalog = [self.target]
        with _fake_gateway(
            catalog=catalog, occupant_cards=[(1, self.target)], loaded={self.variant_id}
        ):
            payload = self._rewrite_and_book(catalog)
        self.assertEqual(payload["model"], self.variant_id)
        self.state.request_started.assert_called_once_with(TARGET_ID)
        for call in self.state.request_started.call_args_list:
            self.assertNotIn(self.variant_id, call.args)

    def test_a_variant_route_skips_the_admission_gate(self):
        """A published variant on a free card is a load that needs no admission.

        Asking the gate first let it retire the pinned card's occupant to satisfy a
        pin that the free card had already made unnecessary.
        """
        catalog = [self.target]
        with _fake_gateway(
            catalog=catalog, occupant_cards=[(1, self.target)], loaded={self.variant_id}
        ):
            self._rewrite_and_book(catalog)
        self.handler._reject_if_gpu_busy.assert_not_called()

    def test_the_gate_still_runs_when_no_variant_was_selected(self):
        catalog = [self.target]
        with _fake_gateway(catalog=catalog, occupant_cards=[(1, self.target)], loaded=set()):
            payload = self._rewrite_and_book(catalog)
        self.assertEqual(payload["model"], TARGET_ID)
        self.handler._reject_if_gpu_busy.assert_called_once()


class RouteSwitchWiringTest(unittest.TestCase):
    """The forwarded payload may carry a __gpuN id; the bookkeeping may not.

    Both ids are plain strings, so nothing but this ordering stops the variant id
    from reaching get_catalog_model_process, _gpu_occupants or the affinity maps,
    where it would silently resolve to nothing because it is not in the catalog.
    """

    @staticmethod
    def _handler_source(method: str) -> str:
        source = inspect.getsource(_cli_impl.start_ctx_metadata_server)
        return textwrap.dedent(source[source.index(method) :])

    @classmethod
    def _method_body(cls, definition: str, next_definition: str | None) -> str:
        body = cls._handler_source(definition)
        if next_definition is None:
            return body
        return body[: body.index(next_definition)]

    def test_the_proxy_payload_carries_the_route_id_and_the_bookkeeping_the_base(self):
        body = self._method_body("def _proxy_request(self,", None)
        self.assertIn("select_model_route(", body)
        self.assertIn('proxy_payload["model"] = route_id', body)
        self.assertLess(
            body.index('proxy_payload["model"] = route_id'),
            body.index("REPLICA_ROUTER_STATE.request_started(upstream_model_name)"),
            "the payload must be rewritten before the request is booked",
        )
        self.assertNotIn("request_started(route_id", body)
        self.assertNotIn("request_finished(route_id", body)

    def test_the_ollama_handler_forwards_the_route_id_only(self):
        body = self._method_body("def _handle_ollama_chat(self):", "\n        def ")
        self.assertIn("select_model_route(", body)
        self.assertIn('"model": upstream_route_id,', body)
        self.assertNotIn("request_started(upstream_route_id", body)
        self.assertNotIn("request_finished(upstream_route_id", body)
        self.assertIn("REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)", body)


class _HandlerRouteSwitchMixin(_RouterStateMixin):
    """One handler's shipped resolve -> gate -> forward block, exec'd for real.

    Every remaining handler nests its own resolver call, its own admission gate and
    its own payload assignment inside the ``start_ctx_metadata_server`` closure, so
    the tests slice those exact lines out of the real module and exec them against
    the real resolver. Nothing but shipped code can therefore put a ``__gpuN`` id
    into the forwarded payload, and the booking that follows still has to name the
    catalog ``model_id``.

    Subclasses declare where their handler starts, where it ends, which api_style its
    gate uses, and which extra names its own payload slice reads.
    """

    HANDLER = ""
    NEXT = ""
    API_STYLE = ""
    FORWARD_FROM = ""
    FORWARD_TO = ""
    FORWARDED = "upstream_payload"
    BOOKING_GUARD = "if int(DEFAULT_MAX_CONCURRENT_PER_MODEL) <= 0 and upstream_model_name:"

    def setUp(self):
        super().setUp()
        self.reset_placement_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = _big_target_model(self.root)
        self.variant_id = f"{TARGET_ID}__gpu1"
        self.state = mock.Mock()
        self.calls: list[str] = []

    def _extra_scope(self) -> dict[str, object]:
        """Names the handler's own payload slice reads besides the route id."""
        return {}

    @staticmethod
    def _block(source: str, first: str, last: str) -> str:
        """The handler's own lines from ``first`` up to (not including) ``last``."""
        start = source.rindex("\n", 0, source.index(first)) + 1
        end = source.rindex("\n", 0, source.index(last, start)) + 1
        return textwrap.dedent(source[start:end])

    @staticmethod
    def _statement_after(source: str, anchor: str) -> str:
        """The single line following ``anchor``, dedented."""
        start = source.index("\n", source.index(anchor)) + 1
        return textwrap.dedent(source[start : source.index("\n", start) + 1])

    def _body(self) -> str:
        source = inspect.getsource(_cli_impl.start_ctx_metadata_server)
        body = source[source.index(self.HANDLER) :]
        return body[: body.index(self.NEXT)] if self.NEXT else body

    def _forward_slice(self, body: str) -> str:
        if self.FORWARD_TO:
            return self._block(body, self.FORWARD_FROM, self.FORWARD_TO)
        return textwrap.dedent(body[body.index(self.FORWARD_FROM) :].split("\n", 1)[0] + "\n")

    def _run(self, catalog, *, upstream_model_name=TARGET_ID, is_replica_request=False, resolver=None):
        """Exec the handler's own block and return the payload it would forward."""
        body = self._body()
        # The handler assigns these two itself further up; seeding them here makes them
        # locals of the probe, which the shipped _spill reset then rebinds. The block
        # under test stays the handler's own lines.
        block = (
            "upstream_model_name = _initial_upstream_model_name\n"
            "is_replica_request = _initial_is_replica_request\n"
        ) + "".join(
            (
                self._block(body, "upstream_route_id = upstream_model_name", "if _blocked:"),
                self._forward_slice(body),
                self._statement_after(body, self.BOOKING_GUARD),
                f"return {self.FORWARDED}\n",
            )
        )
        payload = {"model": upstream_model_name}
        self.handler = mock.Mock()
        self.handler._reject_if_gpu_busy.return_value = False
        self.handler._reject_if_concurrent_limit_exceeded.return_value = (False, None)
        real_resolver = resolver or _cli_impl.select_model_route
        calls = self.calls

        def _recording_resolver(*resolver_args, **resolver_kwargs):
            calls.append("resolve")
            return real_resolver(*resolver_args, **resolver_kwargs)

        scope = {
            "_initial_upstream_model_name": upstream_model_name,
            "_initial_is_replica_request": is_replica_request,
            "select_model_route": _recording_resolver,
            "model_name": TARGET_ID,
            "affinity_key": "",
            "catalog": catalog,
            "args": argparse.Namespace(config=Path("/dev/null"), public_port=11436),
            "client_host": "127.0.0.1",
            "payload": payload,
            "is_replica_model_id": _cli_impl.is_replica_model_id,
            "REPLICA_ROUTER_STATE": self.state,
            "self": self.handler,
            **self._extra_scope(),
        }
        # Wrapped in a function because every refusal path is a `return`, which is
        # not legal at module level.
        exec(compile("def _probe():\n" + textwrap.indent(block, "    "), self.HANDLER, "exec"), scope)
        return scope["_probe"]()

    def _booked(self) -> list[str]:
        return [call.args[0] for call in self.state.request_started.call_args_list]

    def _busy_pin_gateway(self, *, published=(1,)):
        """The reported production incident: the pinned card holds another model and
        the other card is empty."""
        occupant = _make_model(OCCUPANT_ID, str(self.root / f"{OCCUPANT_ID}.safetensors"))
        catalog = [self.target, occupant]
        return catalog, _fake_gateway(
            catalog=catalog,
            occupant_cards=[(0, occupant)],
            published=published,
            free={0: CARD_BUSY_FREE_MIB, 1: CARD_FITS_FREE_MIB},
        )

    # -- 1: a published variant on a free card ------------------------------
    def test_a_published_variant_reaches_the_payload_and_the_gate_is_skipped(self):
        catalog, gateway = self._busy_pin_gateway()
        with gateway:
            payload = self._run(catalog)
        self.assertEqual(payload["model"], self.variant_id)
        self.handler._reject_if_gpu_busy.assert_not_called()
        self.assertEqual(self.calls, ["resolve"], "the switch must decide before the gate is asked")

    def test_the_booking_still_names_the_catalog_model_id(self):
        catalog, gateway = self._busy_pin_gateway()
        with gateway:
            self._run(catalog)
        self.assertEqual(self._booked(), [TARGET_ID])

    # -- 2: no variant selected, the gate still runs ------------------------
    def test_the_gate_still_runs_when_the_switch_cannot_help(self):
        # No published __gpuN route and a pinned card that is busy: the base id is
        # the only route there is, so dropping the gate would admit a load whose
        # occupant was never retired.
        catalog, gateway = self._busy_pin_gateway(published=())
        with gateway:
            payload = self._run(catalog)
        self.assertEqual(payload["model"], TARGET_ID)
        self.handler._reject_if_gpu_busy.assert_called_once()
        self.assertEqual(
            self.handler._reject_if_gpu_busy.call_args.args,
            (TARGET_ID, catalog),
        )
        self.assertEqual(
            self.handler._reject_if_gpu_busy.call_args.kwargs["api_style"],
            self.API_STYLE,
        )
        self.assertEqual(self._booked(), [TARGET_ID])

    def test_a_free_pinned_card_still_goes_through_the_gate(self):
        with _fake_gateway(catalog=[self.target]):
            payload = self._run([self.target])
        self.assertEqual(payload["model"], TARGET_ID)
        self.handler._reject_if_gpu_busy.assert_called_once()

    # -- 3: a replica request is untouched ---------------------------------
    def test_a_replica_request_is_never_rewritten_and_never_admitted(self):
        catalog, _gateway = self._busy_pin_gateway()
        sentinel = f"{TARGET_ID}__gpu0"
        resolver = mock.Mock(return_value=(sentinel, "variant_gpu0"))
        with mock.patch.object(_cli_impl, "select_model_route", resolver):
            payload = self._run(catalog, upstream_model_name=REPLICA_ID, is_replica_request=True, resolver=resolver)
        resolver.assert_not_called()
        self.assertEqual(payload["model"], REPLICA_ID)
        self.handler._reject_if_gpu_busy.assert_not_called()
        self.assertEqual(self._booked(), [REPLICA_ID])

    # -- static invariants --------------------------------------------------
    def test_only_the_forwarded_payload_ever_carries_the_route_id(self):
        body = self._body()
        self.assertIn("select_model_route(", body)
        self.assertLess(
            body.index("select_model_route("),
            body.index("self._reject_if_gpu_busy("),
            "a gate ahead of the resolver retires the pinned card's occupant for a "
            "load a free card already made unnecessary",
        )
        self.assertNotIn("request_started(upstream_route_id", body)
        self.assertNotIn("request_finished(upstream_route_id", body)
        self.assertIn("REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)", body)


class OpenAIChatCompletionsRouteSwitchWiringTest(_HandlerRouteSwitchMixin, unittest.TestCase):
    """The primary OpenAI endpoint, with its vision/image pre-selection intact."""

    HANDLER = "def _handle_openai_chat_completions(self):"
    NEXT = "def _handle_openai_responses(self):"
    API_STYLE = "openai"
    FORWARD_FROM = "upstream_payload = dict(payload)"
    FORWARD_TO = 'upstream_payload["messages"] = messages'

    def _extra_scope(self) -> dict[str, object]:
        # A refusal here has to release the conversation token before it returns.
        return {
            "messages": [{"role": "user", "content": "hi"}],
            "conversation_token": mock.Mock(),
            "CONVERSATION_SWITCH_STATE": mock.Mock(),
        }

    def test_a_vision_request_keeps_its_target_and_never_consults_the_switch(self):
        # route_image_request_to_vision picks the instance carrying --mmproj and the
        # handler marks that target as a replica request, so the switch must leave it
        # alone even while a free card is advertising a variant of the same base.
        catalog, _gateway = self._busy_pin_gateway()
        resolver = mock.Mock(return_value=(f"{TARGET_ID}__gpu0", "variant_gpu0"))
        with mock.patch.object(_cli_impl, "select_model_route", resolver):
            payload = self._run(catalog, upstream_model_name=REPLICA_ID, is_replica_request=True, resolver=resolver)
        resolver.assert_not_called()
        self.assertEqual(payload["model"], REPLICA_ID)
        self.handler._reject_if_gpu_busy.assert_not_called()
        self.assertEqual(self._booked(), [REPLICA_ID])

    def test_the_image_shortcuts_are_still_keyed_on_vision_target(self):
        body = self._body()
        self.assertIn(
            "if vision_target is not None:\n                upstream_model_name = vision_target\n                is_replica_request = True",
            body,
        )
        self.assertIn(
            "if model_entry is not None and vision_target is None and upstream_model_name:\n                reconcile_text_instance_mmproj(",
            body,
        )
        self.assertIn(
            "if _messages_include_images(messages) and (model_entry is None or not (_has_configured_mmproj_runtime(model_entry) or _model_is_vllm_multimodal(model_entry))):",
            body,
        )


class OpenAIResponsesRouteSwitchWiringTest(_HandlerRouteSwitchMixin, unittest.TestCase):
    """The Responses facade: both its passthrough and its chat-compat payload."""

    HANDLER = "def _handle_openai_responses(self):"
    NEXT = "def _handle_ollama_generate(self):"
    API_STYLE = "openai"
    FORWARD_FROM = '_passthrough_payload = {**payload, "model": upstream_route_id}'
    FORWARDED = "_passthrough_payload"

    def test_every_forwarding_site_carries_the_route_id(self):
        body = self._body()
        self.assertNotIn('"model": upstream_model_name', body)
        self.assertIn('_passthrough_payload = {**payload, "model": upstream_route_id}', body)
        self.assertEqual(
            re.findall(r"_responses_payload_to_chat_payload\(\n\s+payload,\n\s+(upstream_\w+),", body),
            ["upstream_route_id", "upstream_route_id"],
            "the chat-compat payload and the internal tool_search round have to follow "
            "the route, not the pinned card",
        )


class OllamaGenerateRouteSwitchWiringTest(_HandlerRouteSwitchMixin, unittest.TestCase):
    """``/api/generate``: the non-streaming Ollama prompt shape."""

    HANDLER = "def _handle_ollama_generate(self):"
    NEXT = "def _handle_ollama_embeddings(self):"
    API_STYLE = "ollama"
    FORWARD_FROM = "upstream_payload = {"
    FORWARD_TO = 'endpoint = "/v1/chat/completions"'

    def _extra_scope(self) -> dict[str, object]:
        # The forwarded payload dict names the messages it was built from.
        return {"upstream_messages": [{"role": "user", "content": "hi"}]}

    def test_the_image_shortcuts_are_still_keyed_on_vision_target(self):
        body = self._body()
        self.assertIn(
            "if vision_target is not None:\n                upstream_model_name = vision_target\n                is_replica_request = True",
            body,
        )
        self.assertIn(
            "if model_entry is not None and vision_target is None and upstream_model_name:\n                reconcile_text_instance_mmproj(",
            body,
        )


class OllamaEmbeddingsRouteSwitchWiringTest(_HandlerRouteSwitchMixin, unittest.TestCase):
    """``/api/embeddings``: no vision path, one inline payload dict."""

    HANDLER = "def _handle_ollama_embeddings(self):"
    NEXT = ""
    API_STYLE = "ollama"
    FORWARD_FROM = 'upstream_payload = {"model": upstream_route_id, "input": text_input}'

    def _extra_scope(self) -> dict[str, object]:
        # The embeddings payload dict names the request text it forwards.
        return {"text_input": "hello"}


VICTIM_VARIANT = f"{OCCUPANT_ID}__gpu1"
VICTIM_REPLICA = f"{OCCUPANT_ID}__replica_0"
VICTIM_PID = 5150
OTHER_VICTIM_PID = 5151
# 3000 MiB free on the card the target is pinned to, against a target that needs
# NEEDED_MIB + 1024 per-GPU overhead + 2048 safety, so only a victim holding more
# than ROOM_SHORTFALL_MIB can make the load fit.
CARD_BUSY_FREE_MIB = 3000.0
ROOM_SHORTFALL_MIB = 23117.0 - CARD_BUSY_FREE_MIB
VICTIM_MIB = ROOM_SHORTFALL_MIB + 1000.0
TOO_SMALL_VICTIM_MIB = 4000.0


class EvictionRankingTest(unittest.TestCase):
    """Which route llama-swap may throw away first.

    A ``__gpuN`` per-card variant exists only to park a model on a card its base does
    not own, so it is the most disposable route in the system and has to rank below a
    replica, which in turn ranks below the base every client asks for by name.
    llama-swap evicts the cheapest entry first, so the ordering is the policy.
    """

    BASE = "big-model"
    REPLICA = f"{BASE}__replica_0"
    VARIANT = f"{BASE}__gpu1"
    SIZE_MIB = 20000.0

    @staticmethod
    def _entry(model_id, gpu_set, size_mib, *, small=False):
        return {
            "id": model_id,
            "gpu_set": list(gpu_set),
            "is_embedding": False,
            "is_small": small,
            "size_mib": size_mib,
        }

    def _costs(self, entries):
        matrix = R._calculate_llama_swap_matrix(entries)
        var_of = {model_id: var for var, model_id in matrix["vars"].items()}
        return matrix, {
            model_id: matrix["evict_costs"][var_of[model_id]] for model_id in var_of
        }

    @staticmethod
    def _co_declared(matrix, first, second):
        var_of = {model_id: var for var, model_id in matrix["vars"].items()}
        pair = {var_of[first], var_of[second]}
        return any(
            pair <= set(expression.replace("+", " ").split())
            for expression in matrix["sets"].values()
        )

    def test_a_variant_is_cheapest_then_a_replica_then_the_base(self):
        _matrix, costs = self._costs(
            [
                self._entry(self.BASE, [0], self.SIZE_MIB),
                self._entry(self.REPLICA, [1], self.SIZE_MIB),
                self._entry(self.VARIANT, [1], self.SIZE_MIB),
            ]
        )
        self.assertEqual(costs[self.BASE], 20000)
        self.assertEqual(costs[self.REPLICA], 10000)
        self.assertEqual(costs[self.VARIANT], 5000)
        self.assertLess(costs[self.VARIANT], costs[self.REPLICA])
        self.assertLess(costs[self.REPLICA], costs[self.BASE])

    def test_no_tier_prices_a_tiny_route_at_zero(self):
        _matrix, costs = self._costs(
            [
                self._entry(self.BASE, [0], 1.0),
                self._entry(self.REPLICA, [1], 1.0),
                self._entry(self.VARIANT, [2], 1.0),
            ]
        )
        self.assertEqual(
            sorted(costs.values()),
            [1, 1, 1],
            "a cost of 0 would win every eviction, including the only one available",
        )

    def test_a_variant_sharing_its_bases_card_is_declared_mutually_exclusive(self):
        matrix = R._calculate_llama_swap_matrix(
            [
                self._entry(self.BASE, [0], self.SIZE_MIB),
                self._entry(f"{self.BASE}__gpu0", [0], self.SIZE_MIB),
            ]
        )
        self.assertFalse(
            self._co_declared(matrix, self.BASE, f"{self.BASE}__gpu0"),
            "two copies of one model on one card is the OOM the per-card routes exist "
            "to avoid, so they must never be declared loadable together",
        )

    def test_a_variant_on_another_card_is_declared_together_with_its_base(self):
        matrix = R._calculate_llama_swap_matrix(
            [
                self._entry(self.BASE, [0], self.SIZE_MIB),
                self._entry(self.VARIANT, [1], self.SIZE_MIB),
            ]
        )
        self.assertTrue(
            self._co_declared(matrix, self.BASE, self.VARIANT),
            "a variant on a card its base does not own is exactly the case this whole "
            "feature exists for, so asking for one must not evict the other",
        )


class RenderEvictionCostTest(unittest.TestCase):
    """The published routes must not move; only the price of losing one may."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "config.yaml"
        # Above the small-model threshold, so the matrix gives it its own group and
        # gpu_route_variants publishes a route per card for it.
        weights = self.root / "big.safetensors"
        with open(weights, "wb") as handle:
            handle.truncate(6 * 1024 * 1024 * 1024)
        self.catalog = [_make_model(TARGET_ID, str(weights))]

    def _render(self):
        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            R.render_llamaswap_config(
                self.catalog, self.config_path, SERVER, 11436, 300, {}, {}
            )
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def _legacy_cost(self, size_mib, model_id):
        return max(1, int(size_mib * (0.5 if R.is_replica_model_id(model_id) else 1.0)))

    def test_the_render_moves_the_evict_costs_and_nothing_else(self):
        current = self._render()
        with mock.patch.object(R, "route_evict_cost", side_effect=self._legacy_cost):
            previous = self._render()
        self.assertEqual(sorted(previous), sorted(current))
        for key in sorted(set(previous) - {"matrix"}):
            self.assertEqual(previous[key], current[key], key)
        for key in ("vars", "sets"):
            self.assertEqual(previous["matrix"][key], current["matrix"][key], key)
        self.assertNotEqual(
            previous["matrix"]["evict_costs"],
            current["matrix"]["evict_costs"],
        )

    def test_every_published_variant_prices_below_the_base_it_copies(self):
        rendered = self._render()
        for card in (0, 1):
            self.assertIn(f"{TARGET_ID}__gpu{card}", rendered["models"])
        var_of = {model_id: var for var, model_id in rendered["matrix"]["vars"].items()}
        costs = rendered["matrix"]["evict_costs"]
        for card in (0, 1):
            self.assertLess(
                costs[var_of[f"{TARGET_ID}__gpu{card}"]], costs[var_of[TARGET_ID]]
            )


# The card capacity an operator declares on this host's 2x4090 pair, and the
# cushion two copies of one model have to leave room for.
CARD_CAPACITY_MIB = 24564
SHARING_SAFETY_MIB = 2048


@contextlib.contextmanager
def _no_live_gpu_state():
    """Tripwires for every probe that would make a render depend on live VRAM.

    A replica's card is decided when config.yaml is written, so occupancy, free VRAM
    or a running process must not reach that decision: it would make two renders of
    one catalog disagree and the file change under a running llama-swap.
    """
    probes = {
        name: mock.Mock(side_effect=AssertionError(f"{name} must not decide placement"))
        for name in (
            "_query_gpu_memory_snapshot_cached",
            "get_gpu_process_memory_by_pid",
            "get_llama_server_processes",
        )
    }
    with contextlib.ExitStack() as stack:
        for name, probe in probes.items():
            stack.enter_context(mock.patch.object(_cli_impl, name, probe))
        yield probes


class SharedCardReplicaTest(unittest.TestCase):
    """A replica may share a card, but only while the copies on it fit.

    With every card already a base - four big models over two cards - the exclusive
    pool was empty and no model got a replica route at all, so `replicas.enabled`
    published nothing. There are now three tiers: a card no base owns, then the
    replica's *own* base's card charged two copies, and only then a neighbour's card
    charged this replica's single share, because its base is not over there. So "two
    copies do not fit" no longer means "no replica": it means "not on its own base's
    card". What still gets nothing is a model whose own share does not fit a card.
    """

    HEAVY_A_MIB = 16000
    HEAVY_B_MIB = 14650
    FITS_MIB = 5000
    DOES_NOT_FIT_MIB = 12000
    #: One whole process is already over a card, so no tier can admit it: this is the
    #: live catalog's refused shape.
    TOO_BIG_FOR_ONE_CARD_MIB = 26704

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # The memo is keyed on catalog identity, and two tests here build different
        # catalogs under the same model ids, so it must not survive the test.
        R._MODEL_GPU_ASSIGNMENT_MEMO["key"] = None

    def _weights(self, model_id: str, mib: int):
        path = self.root / f"{model_id}.safetensors"
        with open(path, "wb") as handle:
            handle.truncate(mib * 1024 * 1024)
        return _make_model(model_id, str(path))

    def _catalog(self, shares_mib: int):
        # heavy-a and heavy-b between them own both cards, so the only card the
        # third model can put a second copy on is the one its own base is pinned to.
        return [
            self._weights("heavy-a", self.HEAVY_A_MIB),
            self._weights("heavy-b", self.HEAVY_B_MIB),
            self._weights("shares", shares_mib),
        ]

    @staticmethod
    def _global_cfg(shares_mib: int) -> dict[str, object]:
        return {
            "enabled": True,
            "max": 1,
            "placement": "exclusive_gpus",
            "gpu_capacity_mib": CARD_CAPACITY_MIB,
            "safety_vram_mib": SHARING_SAFETY_MIB,
        }

    def _replica_routes(self, catalog, global_cfg) -> dict[str, list[int]]:
        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            R._MODEL_GPU_ASSIGNMENT_MEMO["key"] = None
            return {
                model.model_id: list(gpu_set or [])
                for model, _public_base, gpu_set in R.iter_catalog_with_replicas(
                    catalog, global_cfg
                )
            }

    def _matrix_view(self, catalog, global_cfg) -> dict[str, tuple[int, frozenset[str]]]:
        """Each route's eviction price and the routes it may be loaded next to.

        Matrix set names are positional, so the group a route lands in is compared as
        the set of its peers rather than by name.
        """
        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            R._MODEL_GPU_ASSIGNMENT_MEMO["key"] = None
            entries = R.iter_catalog_with_replicas(catalog, global_cfg)
            assignment = R.assign_model_gpu_sets(catalog, 2)
        matrix = R._calculate_llama_swap_matrix(
            [
                {
                    "id": model.model_id,
                    "gpu_set": list(
                        gpu_set or assignment.get(model.model_id) or [0, 1]
                    ),
                    "is_embedding": R._is_embedding_model(model),
                    "is_small": R._is_small_model(model),
                    "size_mib": R._get_model_size_mib(model),
                }
                for model, _public_base, gpu_set in entries
            ]
        )
        var_of = {var: model_id for var, model_id in matrix["vars"].items()}
        var_id_of = {model_id: var for var, model_id in matrix["vars"].items()}
        groups = [
            {var for var in str(expression).split(" & ") if var in var_of}
            for expression in matrix["sets"].values()
        ]
        return {
            model_id: (
                matrix["evict_costs"][var_id_of[model_id]],
                frozenset(
                    other
                    for group in groups
                    if model_id in group
                    for other in group
                    if other != model_id
                ),
            )
            for model_id in matrix["vars"].values()
        }

    def test_two_copies_that_fit_get_a_replica_on_the_bases_own_card(self):
        catalog = self._catalog(self.FITS_MIB)
        routes = self._replica_routes(catalog, self._global_cfg(self.FITS_MIB))
        self.assertIn("shares__replica_0", routes)
        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            R._MODEL_GPU_ASSIGNMENT_MEMO["key"] = None
            self.assertEqual(routes["shares__replica_0"], R.assign_model_gpu_sets(catalog, 2)["shares"])

    def test_two_copies_that_do_not_fit_still_get_a_replica_on_a_neighbours_card(self):
        catalog = self._catalog(self.DOES_NOT_FIT_MIB)
        routes = self._replica_routes(catalog, self._global_cfg(self.DOES_NOT_FIT_MIB))
        self.assertIn("shares", routes)
        # The two-copy rule only closes the model's own base's card; this one's
        # footprint is still under a whole card, so it takes a neighbour's card
        # instead, charged a single share.
        self.assertEqual(routes["shares__replica_0"], [1])
        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            R._MODEL_GPU_ASSIGNMENT_MEMO["key"] = None
            assignment = R.assign_model_gpu_sets(catalog, 2)
        self.assertEqual(assignment["shares"], [0])
        self.assertNotIn(0, routes["shares__replica_0"])

    def test_a_model_too_big_for_one_card_gets_no_replica_at_any_tier(self):
        catalog = self._catalog(self.TOO_BIG_FOR_ONE_CARD_MIB)
        routes = self._replica_routes(catalog, self._global_cfg(self.TOO_BIG_FOR_ONE_CARD_MIB))
        self.assertIn("shares", routes)
        self.assertNotIn("shares__replica_0", routes)
        model = next(m for m in catalog if m.model_id == "shares")
        runtime = float(models_mod.estimate_model_runtime_mib(model))
        self.assertGreater(runtime + SHARING_SAFETY_MIB, CARD_CAPACITY_MIB)

    def test_the_replica_routes_are_a_pure_function_of_the_catalog(self):
        catalog = self._catalog(self.FITS_MIB)
        first = self._replica_routes(catalog, self._global_cfg(self.FITS_MIB))
        with _no_live_gpu_state() as probes:
            second = self._replica_routes(catalog, self._global_cfg(self.FITS_MIB))
        self.assertEqual(first, second)
        self.assertEqual(first["shares__replica_0"], [0])
        for probe in probes.values():
            probe.assert_not_called()

    def test_the_refusal_names_the_capacity_shortfall(self):
        # A model that gets a neighbour's card never reaches the refusal path, so
        # the refusal is pinned on one whose own share cannot fit a card at all.
        catalog = self._catalog(self.TOO_BIG_FOR_ONE_CARD_MIB)
        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            lines = R.summarize_configured_replicas(
                catalog, self._global_cfg(self.TOO_BIG_FOR_ONE_CARD_MIB)
            )
        refusal = next(line for line in lines if line.startswith("shares:"))
        needed = 2 * self.TOO_BIG_FOR_ONE_CARD_MIB + SHARING_SAFETY_MIB
        self.assertIn("replicas enabled but no replica card", refusal)
        self.assertIn(str(needed), refusal)
        self.assertIn(str(needed - CARD_CAPACITY_MIB), refusal)
        self.assertIn(f"gpu_capacity_mib={CARD_CAPACITY_MIB}", refusal)
        self.assertNotIn("no GPU set generated", refusal)

    def test_the_refusal_quotes_raw_weights_and_not_the_footprint_it_decided_on(self):
        # `_replica_capacity_refusal` still reads the weights while every capacity
        # decision reads the runtime footprint, so the shortfall it quotes is about
        # 3.5 GiB smaller than the one that was actually applied. Pinned as it
        # behaves, deliberately: the message is what an operator acts on, and this
        # is the discrepancy to fix, not to enshrine.
        catalog = self._catalog(self.TOO_BIG_FOR_ONE_CARD_MIB)
        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            lines = R.summarize_configured_replicas(
                catalog, self._global_cfg(self.TOO_BIG_FOR_ONE_CARD_MIB)
            )
        refusal = next(line for line in lines if line.startswith("shares:"))
        weights_mib = R._get_model_size_mib(catalog[2])
        runtime_mib = float(models_mod.estimate_model_runtime_mib(catalog[2]))
        self.assertEqual(weights_mib, float(self.TOO_BIG_FOR_ONE_CARD_MIB))
        self.assertGreater(runtime_mib, weights_mib)
        self.assertIn(f"two copies of {int(weights_mib)} MiB", refusal)
        self.assertIn(str(int(2 * weights_mib + SHARING_SAFETY_MIB)), refusal)
        self.assertNotIn(str(int(2 * runtime_mib + SHARING_SAFETY_MIB)), refusal)

    def test_a_model_that_gains_nothing_keeps_its_vars_costs_and_peers(self):
        # `oversized` is in the catalog so "gains nothing" is a real claim: with
        # only the three fittable models every one of them now gains a replica.
        catalog = self._catalog(self.FITS_MIB) + [
            self._weights("oversized", self.TOO_BIG_FOR_ONE_CARD_MIB)
        ]
        before = self._matrix_view(catalog, self._global_cfg(self.FITS_MIB) | {"enabled": False})
        after = self._matrix_view(catalog, self._global_cfg(self.FITS_MIB))
        self.assertEqual(
            set(after) - set(before),
            {"heavy-a__replica_0", "heavy-b__replica_0", "shares__replica_0"},
        )
        self.assertIn("oversized", after)
        self.assertEqual(
            [model_id for model_id in before if model_id not in after], []
        )
        for model_id, (cost, peers) in before.items():
            self.assertEqual(after[model_id][0], cost, model_id)
            # A route that did not exist before can only ever add a peer: something
            # new landed on another card and is now co-loadable with it.
            self.assertLessEqual(peers, after[model_id][1], model_id)
        self.assertEqual(before["oversized"], after["oversized"])


class SharedCardRuntimeFootprintTest(unittest.TestCase):
    """Sharing a card is decided on the process footprint, not on the weights.

    ``_replica_fits_cards`` used to compare raw on-disk weights against a card that
    also has to hold two processes' KV caches and compute buffers, so on a host where
    that ~3.5 GiB gap crossed the threshold it admitted a load that then OOMed. One
    estimator now answers every capacity question, and a model it cannot measure is
    still never admitted.
    """

    #: Two copies of these weights plus the cushion fit the declared card, so the
    #: weights-only rule admitted the replica.
    WEIGHTS_FIT_RUNTIME_DOES_NOT_MIB = 10000
    #: Two copies of these weights plus the runtime overhead both fit.
    BOTH_FIT_MIB = 5000

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _weights(self, model_id: str, mib: int) -> str:
        path = self.root / f"{model_id}.safetensors"
        with open(path, "wb") as handle:
            handle.truncate(mib * 1024 * 1024)
        return str(path)

    @staticmethod
    def _cfg() -> ReplicaConfig:
        return ReplicaConfig(
            enabled=True,
            max=1,
            placement="exclusive_gpus",
            gpu_capacity_mib=CARD_CAPACITY_MIB,
            safety_vram_mib=SHARING_SAFETY_MIB,
        )

    def _shares_with_its_own_base(self, local_path: str) -> bool:
        # One card, held by this model's own base, so the rule under test is the only
        # thing deciding the answer.
        return R._replica_fits_cards(
            _make_model("shares", local_path), self._cfg(), [0], {0}
        )

    def test_weights_that_fit_but_runtime_does_not_are_refused_the_card(self):
        mib = self.WEIGHTS_FIT_RUNTIME_DOES_NOT_MIB
        model = _make_model("shares", self._weights("shares", mib))
        # Precondition: the weights-only rule this replaces would have admitted it.
        self.assertLessEqual(2 * mib + SHARING_SAFETY_MIB, CARD_CAPACITY_MIB)
        runtime = models_mod.estimate_model_runtime_mib(model)
        self.assertIsNotNone(runtime)
        self.assertGreater(2 * float(runtime) + SHARING_SAFETY_MIB, CARD_CAPACITY_MIB)
        self.assertFalse(self._shares_with_its_own_base(model.local_path))

    def test_a_runtime_footprint_that_fits_is_still_admitted(self):
        model = _make_model("shares", self._weights("shares", self.BOTH_FIT_MIB))
        runtime = models_mod.estimate_model_runtime_mib(model)
        self.assertLessEqual(2 * float(runtime) + SHARING_SAFETY_MIB, CARD_CAPACITY_MIB)
        self.assertTrue(self._shares_with_its_own_base(model.local_path))

    def test_a_model_whose_footprint_cannot_be_measured_is_refused(self):
        missing = str(self.root / "not-downloaded.safetensors")
        self.assertIsNone(models_mod.estimate_model_runtime_mib(_make_model("m", missing)))
        self.assertFalse(self._shares_with_its_own_base(missing))

    def test_the_estimator_is_one_object_and_impl_patching_still_reaches_placement_fits(self):
        self.assertIs(
            _cli_impl.estimate_model_runtime_mib, models_mod.estimate_model_runtime_mib
        )
        model = _make_model("target", self._weights("target", 2048))
        with (
            mock.patch.object(
                _cli_impl, "estimate_model_runtime_mib", return_value=12345.0
            ) as patched,
            mock.patch.object(_cli_impl, "_query_gpu_memory_snapshot_cached", return_value={}),
        ):
            fits, required = _cli_impl._placement_fits(model, self._cfg(), [0], {})
        patched.assert_called_once_with(model)
        self.assertEqual(required, 12345.0)
        self.assertTrue(fits)

        # Patching through the package must land on that same name and leave it there.
        # `llamacpp_stack.cli` resolves names it does not own through __getattr__, and
        # mock.patch then deletes instead of restoring them - which __delattr__
        # cascades into _cli_impl, so a name reachable only that way leaves every
        # call site in this module raising NameError once the patch exits.
        self.assertIn("estimate_model_runtime_mib", vars(cli))
        with mock.patch("llamacpp_stack.cli.estimate_model_runtime_mib", return_value=99.0):
            self.assertEqual(_cli_impl.estimate_model_runtime_mib(None), 99.0)
        self.assertIs(
            _cli_impl.estimate_model_runtime_mib, models_mod.estimate_model_runtime_mib
        )

    def test_both_size_callers_measure_through_the_one_helper(self):
        model = _make_model("m", self._weights("m", 1024))
        measured: list[str] = []

        def fake_size(candidate) -> float:
            measured.append(candidate.model_id)
            return 2048.0

        # One patch on the owner: replica resolves the helper off cli.models at call
        # time, so both callers have to be reaching the same measurement.
        with mock.patch.object(
            models_mod, "model_weights_size_mib", side_effect=fake_size
        ):
            runtime = models_mod.estimate_model_runtime_mib(model)
            weights = R._get_model_size_mib(model)
        self.assertEqual(measured, ["m", "m"])
        self.assertGreater(float(runtime), weights)
        # The helper's failure value is None; each caller maps it to its own contract.
        with mock.patch.object(models_mod, "model_weights_size_mib", return_value=None):
            self.assertIsNone(models_mod.estimate_model_runtime_mib(model))
            self.assertEqual(R._get_model_size_mib(model), 0.0)


def _lift(source: str, first: str, last: str) -> str:
    """The lines from the one containing ``first`` up to the one holding ``last``."""
    start = source.rindex("\n", 0, source.index(first)) + 1
    end = source.rindex("\n", 0, source.index(last, start)) + 1
    return textwrap.dedent(source[start:end])


class KeepaliveRouteTest(unittest.TestCase):
    """The keepalive reloads the route that is live, not the base that went away.

    A ``__gpuN`` route holds the same weights as its base, so reissuing the base id
    after an unexpected unload starts a second copy of a model that is already
    resident - which is what left two live copies of one model on two cards.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = _big_target_model(self.root)
        self.variant_id = f"{TARGET_ID}__gpu1"
        self.catalog = [self.target]

    def _keepalive(self, events: mock.Mock) -> list[str]:
        """Run the guard's reload half and report the route ids it asked for.

        ``start_unexpected_unload_guard`` runs a daemon loop with a sleep in it, so
        the statements that decide which route to reload are lifted out and run
        here against the real resolver: only the probes and the catalog read are
        stubbed.
        """
        touched: list[str] = []

        def _touch(route_id, host, port):
            touched.append(route_id)
            return True

        block = _lift(
            inspect.getsource(_cli_impl.start_unexpected_unload_guard),
            "for model_id in sorted(disappeared):",
            "except Exception as exc:",
        )
        activity: dict[str, object] = {}
        scope = {
            "activity": activity,
            "args": argparse.Namespace(config=Path("/dev/null")),
            "catalog": self.catalog,
            "catalog_path": Path("/dev/null"),
            "config_path": Path("/dev/null"),
            "disappeared": {TARGET_ID},
            "host": "127.0.0.1",
            "idle_ttl": 300,
            "last_activity_model_id": TARGET_ID,
            "load_catalog": mock.Mock(return_value=self.catalog),
            "log_api_event": events,
            "log_model_runtime_snapshot": mock.Mock(),
            "now": _monotonic(),
            "port": 11436,
            "select_model_route": _cli_impl.select_model_route,
            "should_reload_after_unexpected_unload": mock.Mock(return_value=(True, 1.0)),
            "state": {"loaded": {TARGET_ID}},
            "time": _cli_impl.time,
            "_touch_model_via_llamaswap": _touch,
            "_try_drop_caches": mock.Mock(return_value=True),
            "get_loaded_catalog_model_ids": mock.Mock(return_value={TARGET_ID}),
            "get_model_activity_snapshot": mock.Mock(return_value=(activity, TARGET_ID)),
        }
        bundle = mock.MagicMock()
        bundle.collect_engine_crash_bundle.return_value = {"pid": OCCUPANT_PID}
        with mock.patch.dict(
            sys.modules, {"llamacpp_stack.cli.crash_bundle": bundle}
        ):
            exec(
                compile("def _probe():\n" + textwrap.indent(block, "    "), "<guard>", "exec"),
                scope,
            )
            scope["_probe"]()
        return touched

    def _run(self, loaded):
        with _fake_gateway(
            catalog=self.catalog,
            occupant_cards=[(1, self.target)],
            loaded=loaded,
        ) as probes:
            return self._keepalive(probes["log_api_event"]), probes["log_api_event"]

    def test_a_live_variant_is_reloaded_instead_of_the_base(self):
        touched, _events = self._run({self.variant_id})
        self.assertEqual(touched, [self.variant_id])
        self.assertNotIn(TARGET_ID, touched)

    def test_a_live_base_is_still_reloaded_by_its_own_id(self):
        touched, _events = self._run({TARGET_ID})
        self.assertEqual(touched, [TARGET_ID])

    def test_the_keepalive_reports_no_anomaly_when_one_route_is_live(self):
        _touched, events = self._run({self.variant_id})
        kinds = [call.args[0] for call in events.call_args_list]
        self.assertNotIn("route_multiple_live_routes", kinds)


@contextlib.contextmanager
def _card_under_pressure(
    occupants,
    *,
    loaded=(),
    free=CARD_BUSY_FREE_MIB,
    unload_ok=True,
    unload_error=None,
):
    """Fake every probe the admission path reads, so nothing here needs a GPU.

    occupants is what ``_gpu_occupants`` would report for the target's card, and
    ``loaded`` is the route ids llama-swap reports as running - the only view that can
    tell a ``__gpuN`` variant from the base it shares a weights file with. Yields the
    mocks so a test can read what was unloaded and what was logged.
    """
    events: list[tuple[str, dict]] = []
    unload = mock.Mock(return_value=unload_ok)
    if unload_error is not None:
        unload.side_effect = unload_error
    snapshot = {
        card: {"used_mib": CARD_TOTAL_MIB - free, "free_mib": free, "total_mib": CARD_TOTAL_MIB}
        for card in (0, 1)
    }
    patches = {
        "_gpu_occupants": mock.Mock(return_value={1: list(occupants)}),
        "get_loaded_model_ids": mock.Mock(return_value=set(loaded)),
        "estimate_model_runtime_mib": mock.Mock(return_value=NEEDED_MIB),
        "detect_cuda_device_count": mock.Mock(return_value=2),
        # The planner and effective_model_gpu_set resolve these from the module, not
        # from a handler's local scope, so they have to be faked here too.
        "cached_model_gpu_sets": mock.Mock(return_value={TARGET_ID: [1]}),
        "get_catalog_model_process": mock.Mock(return_value=None),
        "_query_gpu_memory_snapshot_cached": mock.Mock(return_value=snapshot),
        "log_api_event": mock.Mock(
            side_effect=lambda event, data=None, **_kw: events.append((event, data or {}))
        ),
    }
    with contextlib.ExitStack() as stack:
        for name, value in patches.items():
            stack.enter_context(mock.patch.object(_cli_impl, name, value))
        stack.enter_context(
            mock.patch.object(daemon_mod, "unload_replica_via_llamaswap", unload)
        )
        yield type(
            "Probes",
            (),
            {"unload": unload, "events": events, "kinds": [name for name, _ in events]},
        )()


class DisposableRouteEvictionTest(_RouterStateMixin, unittest.TestCase):
    """A load that needs VRAM takes it from a spare copy, never from a base route.

    Two guards make this safe. The candidate has to be a route that exists only to hold
    a spare copy - a ``__gpuN`` variant or a ``__replica_N`` - so the base every client
    asks for by name is never a victim; and it has to be idle, so nothing streaming is
    cancelled to admit the request.
    """

    def setUp(self):
        self.reset_placement_state()
        self.catalog = [_make_model(TARGET_ID, "/models/target.safetensors")]

    def _evict(self):
        return _cli_impl.evict_disposable_route_for_load(
            TARGET_ID, self.catalog, [1], client_host="127.0.0.1", public_port=11436
        )

    def _busy_variant(self, mib=VICTIM_MIB):
        # pgrep names the process after its weights file, which the variant shares with
        # its base, so the occupant arrives as the base id.
        return [(OCCUPANT_ID, VICTIM_PID, mib)]

    def test_a_probe_that_raises_leaves_the_load_refused_and_nothing_escapes(self):
        # The unload is only half the path: the occupant scan itself can fail (nvidia-smi
        # gone, pgrep timing out), and the except handler has to survive that too.
        with _card_under_pressure(
            self._busy_variant(), loaded=(VICTIM_VARIANT,)
        ) as probes:
            with mock.patch.object(
                _cli_impl, "_gpu_occupants", side_effect=OSError("nvidia-smi is gone")
            ):
                admitted = self._evict()
        self.assertFalse(admitted)
        probes.unload.assert_not_called()
        self.assertEqual(
            dict(probes.events)["disposable_route_evict_failed"]["route"],
            "",
            "with no victim chosen there is no route to name, and the handler must say so",
        )

    def test_a_busy_card_gives_up_its_idle_variant_and_the_load_goes_through(self):
        with _card_under_pressure(
            self._busy_variant(), loaded=(VICTIM_VARIANT,)
        ) as probes:
            admitted = self._evict()
        self.assertTrue(admitted)
        self.assertEqual(
            [call.args for call in probes.unload.call_args_list],
            [(VICTIM_VARIANT, "127.0.0.1", 11436)],
            "the unload must name the variant route, which is the id llama-swap knows it by",
        )
        event = dict(probes.events)["disposable_route_evicted"]
        self.assertEqual(event["route"], VICTIM_VARIANT)
        self.assertEqual(event["gpu"], 1)
        self.assertEqual(event["freed_mib"], VICTIM_MIB)
        self.assertTrue(event["admitted"])

    def test_a_base_route_on_the_card_is_never_sacrificed(self):
        # The safety property: the route every client asks for by name is not a
        # candidate at all, so there is nothing to unload and the load stays refused.
        with _card_under_pressure(
            [(OCCUPANT_ID, VICTIM_PID, VICTIM_MIB)], loaded=(OCCUPANT_ID,)
        ) as probes:
            admitted = self._evict()
        self.assertFalse(admitted)
        probes.unload.assert_not_called()
        self.assertEqual(probes.events, [], "nothing was given up, so nothing to report")

    def test_a_variant_with_requests_in_flight_is_left_alone(self):
        # A __gpuN route is booked under its base id: the handler rewrites the payload
        # but not the bookkeeping, so the base's counter is the traffic on that card.
        _cli_impl.REPLICA_ROUTER_STATE.base_in_flight[OCCUPANT_ID] = 1
        with _card_under_pressure(
            self._busy_variant(), loaded=(VICTIM_VARIANT,)
        ) as probes:
            admitted = self._evict()
        self.assertFalse(admitted)
        probes.unload.assert_not_called()

    def test_a_streaming_variant_defers_to_an_idle_replica_on_the_same_card(self):
        _cli_impl.REPLICA_ROUTER_STATE.base_in_flight[OCCUPANT_ID] = 1
        _cli_impl.REPLICA_ROUTER_STATE.records[VICTIM_REPLICA] = ReplicaRecord(
            base_model_id="some-other-model",
            replica_model_id=VICTIM_REPLICA,
            gpu_set=[1],
            in_flight=0,
        )
        with _card_under_pressure(
            self._busy_variant() + [(VICTIM_REPLICA, OTHER_VICTIM_PID, VICTIM_MIB)],
            loaded=(VICTIM_VARIANT, VICTIM_REPLICA),
        ) as probes:
            admitted = self._evict()
        self.assertTrue(admitted)
        self.assertEqual(
            [call.args[0] for call in probes.unload.call_args_list], [VICTIM_REPLICA]
        )

    def test_a_busy_replica_is_never_sacrificed_for_its_own_busy_base(self):
        _cli_impl.REPLICA_ROUTER_STATE.records[VICTIM_REPLICA] = ReplicaRecord(
            base_model_id=OCCUPANT_ID,
            replica_model_id=VICTIM_REPLICA,
            gpu_set=[1],
            in_flight=2,
        )
        with _card_under_pressure(
            [(VICTIM_REPLICA, VICTIM_PID, VICTIM_MIB)], loaded=(VICTIM_REPLICA,)
        ) as probes:
            admitted = self._evict()
        self.assertFalse(admitted)
        probes.unload.assert_not_called()

    def test_an_unload_that_raises_leaves_the_load_refused_and_nothing_escapes(self):
        with _card_under_pressure(
            self._busy_variant(),
            loaded=(VICTIM_VARIANT,),
            unload_error=RuntimeError("llama-swap is gone"),
        ) as probes:
            admitted = self._evict()
        self.assertFalse(admitted)
        self.assertEqual(
            dict(probes.events)["disposable_route_evict_failed"]["route"], VICTIM_VARIANT
        )

    def test_an_unload_llama_swap_refuses_does_not_report_a_sacrifice(self):
        with _card_under_pressure(
            self._busy_variant(), loaded=(VICTIM_VARIANT,), unload_ok=False
        ) as probes:
            admitted = self._evict()
        self.assertFalse(admitted)
        self.assertNotIn("disposable_route_evicted", probes.kinds)

    def test_a_victim_too_small_to_help_still_leaves_the_load_refused(self):
        with _card_under_pressure(
            self._busy_variant(TOO_SMALL_VICTIM_MIB), loaded=(VICTIM_VARIANT,)
        ) as probes:
            admitted = self._evict()
        self.assertFalse(admitted)
        self.assertEqual(
            [call.args[0] for call in probes.unload.call_args_list], [VICTIM_VARIANT]
        )
        event = dict(probes.events)["disposable_route_evict_insufficient"]
        self.assertFalse(event["admitted"])
        self.assertEqual(
            event["checks"][0]["free_mib"],
            CARD_BUSY_FREE_MIB + TOO_SMALL_VICTIM_MIB,
            "the answer must be the measured free memory plus exactly what was freed",
        )

    def test_the_gate_forwards_instead_of_answering_503_once_the_card_is_free(self):
        # The end-to-end shape: the gate reaches for a spare copy, gets one, and never
        # reaches the refusal that used to be its only outcome.
        source = inspect.getsource(_cli_impl.start_ctx_metadata_server)
        start = source.index("def _reject_if_gpu_busy(self,")
        end = source.index("def _reject_if_model_loading(self,", start)
        sent: list[tuple[dict, int]] = []
        stub = mock.Mock()
        stub._send_json = lambda payload, status=200: sent.append((payload, status))
        namespace = {
            "args": argparse.Namespace(config=Path("/dev/null"), public_port=11436),
            "client_host": "127.0.0.1",
            "ManagedModel": R.ManagedModel,
            "log_api_event": mock.Mock(),
            "REPLICA_ROUTER_STATE": _cli_impl.REPLICA_ROUTER_STATE,
            "blocking_conflicting_load": _cli_impl.blocking_conflicting_load,
            "plan_runtime_placement": _cli_impl.plan_runtime_placement,
            "retire_idle_occupants": _cli_impl.retire_idle_occupants,
            "apply_repin": _cli_impl.apply_repin,
            "evict_disposable_route_for_load": _cli_impl.evict_disposable_route_for_load,
            "effective_model_gpu_set": _cli_impl.effective_model_gpu_set,
            "get_catalog_model_process": mock.Mock(return_value=None),
            "get_gpu_conflict_message": mock.Mock(return_value="the GPU is busy"),
            "cached_model_gpu_sets": mock.Mock(return_value={TARGET_ID: [1]}),
            "get_model_activity_snapshot": mock.Mock(return_value=({}, None)),
            "recent_activity_blocking_model_switch": mock.Mock(return_value=None),
            "request_looks_like_model_probe": mock.Mock(return_value=False),
            "time": _cli_impl.time,
            "_resolve_model_probe_autoload_config": mock.Mock(return_value={"enabled": False}),
        }
        exec(compile(textwrap.dedent(source[start:end]), "<gate>", "exec"), namespace)
        gate = namespace["_reject_if_gpu_busy"]
        with _card_under_pressure(
            self._busy_variant(), loaded=(VICTIM_VARIANT,)
        ) as probes:
            refused = gate(stub, TARGET_ID, self.catalog, api_style="openai", payload={})
        self.assertFalse(refused)
        self.assertEqual(sent, [], "an admitted request must not receive the 503")
        self.assertEqual(
            [call.args[0] for call in probes.unload.call_args_list], [VICTIM_VARIANT]
        )

    def test_the_gate_still_answers_its_own_503_when_nothing_is_disposable(self):
        source = inspect.getsource(_cli_impl.start_ctx_metadata_server)
        start = source.index("def _reject_if_gpu_busy(self,")
        end = source.index("def _reject_if_model_loading(self,", start)
        conflict = "Cannot load model 'x': the GPU is already in use"
        sent: list[tuple[dict, int]] = []
        stub = mock.Mock()
        stub._send_json = lambda payload, status=200: sent.append((payload, status))
        namespace = {
            "args": argparse.Namespace(config=Path("/dev/null"), public_port=11436),
            "client_host": "127.0.0.1",
            "ManagedModel": R.ManagedModel,
            "log_api_event": mock.Mock(),
            "REPLICA_ROUTER_STATE": _cli_impl.REPLICA_ROUTER_STATE,
            "blocking_conflicting_load": _cli_impl.blocking_conflicting_load,
            "plan_runtime_placement": _cli_impl.plan_runtime_placement,
            "retire_idle_occupants": _cli_impl.retire_idle_occupants,
            "apply_repin": _cli_impl.apply_repin,
            "evict_disposable_route_for_load": _cli_impl.evict_disposable_route_for_load,
            "effective_model_gpu_set": _cli_impl.effective_model_gpu_set,
            "get_catalog_model_process": mock.Mock(return_value=None),
            "get_gpu_conflict_message": mock.Mock(return_value=conflict),
            "cached_model_gpu_sets": mock.Mock(return_value={TARGET_ID: [1]}),
            "get_model_activity_snapshot": mock.Mock(return_value=({}, None)),
            "recent_activity_blocking_model_switch": mock.Mock(return_value=None),
            "request_looks_like_model_probe": mock.Mock(return_value=False),
            "time": _cli_impl.time,
            "_resolve_model_probe_autoload_config": mock.Mock(return_value={"enabled": False}),
        }
        exec(compile(textwrap.dedent(source[start:end]), "<gate>", "exec"), namespace)
        gate = namespace["_reject_if_gpu_busy"]
        with _card_under_pressure(
            [(OCCUPANT_ID, VICTIM_PID, VICTIM_MIB)], loaded=(OCCUPANT_ID,)
        ) as probes:
            refused = gate(stub, TARGET_ID, self.catalog, api_style="openai", payload={})
        self.assertTrue(refused)
        probes.unload.assert_not_called()
        self.assertEqual(len(sent), 1)
        payload, status = sent[0]
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["message"], conflict)


class DisposableEvictionWiringTest(unittest.TestCase):
    """Where the hook sits, and what it is allowed to touch."""

    @staticmethod
    def _gate_body() -> str:
        source = inspect.getsource(_cli_impl.start_ctx_metadata_server)
        start = source.index("def _reject_if_gpu_busy(self,")
        return source[start : source.index("def _reject_if_model_loading(self,", start)]

    def test_making_room_is_tried_after_the_conflict_and_before_the_refusal(self):
        body = self._gate_body()
        self.assertLess(
            body.index("if not gpu_conflict:"), body.index("evict_disposable_route_for_load(")
        )
        self.assertLess(
            body.index("evict_disposable_route_for_load("),
            body.index('log_api_event("model_load_blocked_gpu_busy"'),
            "the hook has to run before the 503 is logged and sent, or a request that "
            "could have been admitted is already refused",
        )

    def test_making_room_reaches_no_config_writer(self):
        # llama-swap runs --watch-config, so ANY config.yaml write tears down every
        # loaded model: a retirement path that rendered the config would manufacture
        # the exact ping-pong this feature removes.
        source = inspect.getsource(_cli_impl.evict_disposable_route_for_load) + inspect.getsource(
            _cli_impl.disposable_route_victims
        )
        for name in (
            "render_llamaswap_config",
            "_write_config_atomic",
            "ensure_internal_route",
            "ensure_replica_route",
            "drop_internal_route",
            "set_instance_mmproj",
            "save_catalog",
        ):
            self.assertNotIn(name, source, name)

    def test_making_room_unloads_through_the_existing_swap_endpoint(self):
        source = inspect.getsource(_cli_impl.evict_disposable_route_for_load)
        self.assertIn("unload_replica_via_llamaswap", source)
        self.assertNotIn("requests.", source)


class _WarmerProbes:
    """Live view over what the warmer read, issued and logged on this tick.

    ``kinds`` and ``by_kind`` are properties, not snapshots: the ticks run after the
    context manager has already yielded, so a value captured at entry time would be
    empty for every assertion about what a tick decided.
    """

    def __init__(self, *, warm, post, events, patches):
        self.warm = warm
        self.post = post
        self.events = events
        self.patches = patches

    @property
    def kinds(self) -> list[str]:
        return [name for name, _ in self.events]

    @property
    def by_kind(self) -> dict[str, dict]:
        return {name: data for name, data in self.events}


class IdleWarmerTest(unittest.TestCase):
    """R2: an idle card keeps a warm copy, by loading a published route and nothing else.

    The warmer's whole safety argument is that it never writes ``config.yaml``. Every
    other rule here exists so that, when it does load, it loads the route a request would
    have loaded anyway: on a card that has genuinely been empty, with the weights already
    readable, with nothing in flight, and at most once per tick.
    """

    WARM_IDLE_AFTER_S = 60.0
    WARM_COOLDOWN_S = 900.0
    WARM_T0 = 1_000_000.0

    def setUp(self):
        # The warmer keeps process-lifetime memory; a leaked idle_since from another test
        # would make a fresh card read as long-idle on its very first tick.
        self._clear_warmer_state()
        self.addCleanup(self._clear_warmer_state)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = _big_target_model(self.root)
        self.variant_id = f"{TARGET_ID}__gpu1"
        self.other_variant_id = f"{TARGET_ID}__gpu0"
        self.args = argparse.Namespace(
            config=Path("/dev/null"),
            catalog=Path("/dev/null"),
            server_config=Path("/dev/null"),
            public_host="127.0.0.1",
            public_port=11436,
        )

    @staticmethod
    def _clear_warmer_state():
        state = _cli_impl._IDLE_WARMER_STATE
        with state.lock:
            state.idle_since.clear()
            state.cooldown_until.clear()
            state.warming.clear()

    @contextlib.contextmanager
    def _warmer(
        self,
        *,
        catalog=None,
        gpu_sets=None,
        published=(),
        loaded=(),
        occupant_cards=(),
        free=None,
        fit=True,
        fit_info=None,
        in_flight=None,
        total_gpus=2,
        sizes=None,
        enabled=True,
        warm_ok=True,
        warm_error=None,
        real_http=False,
    ):
        catalog = [self.target] if catalog is None else list(catalog)
        gpu_sets = {model.model_id: [0] for model in catalog} if gpu_sets is None else gpu_sets
        free_mib = {0: CARD_FITS_FREE_MIB, 1: CARD_FITS_FREE_MIB} if free is None else dict(free)
        per_pid: dict[int, dict[int, float]] = {}
        processes = []
        for index, card in enumerate(occupant_cards):
            pid = OCCUPANT_PID + index
            processes.append(
                {
                    "pid": pid,
                    "cmdline": f"llama-server --model /models/occupant{index} --port {12000 + pid}",
                    "model_path": f"/models/occupant{index}",
                    "port": 12000 + pid,
                }
            )
            per_pid[pid] = {int(card): OCCUPANT_MIB}
        snapshot = {
            gpu: {
                "used_mib": CARD_TOTAL_MIB - float(free_mib.get(gpu, 0.0)),
                "free_mib": float(free_mib.get(gpu, 0.0)),
                "total_mib": CARD_TOTAL_MIB,
            }
            for gpu in range(total_gpus)
        }
        flights = dict(in_flight or {})
        events: list[tuple[str, dict]] = []
        payload = {
            "experimental": {
                "idle_gpu_warmer": {
                    "enabled": enabled,
                    "idle_after_s": self.WARM_IDLE_AFTER_S,
                    "cooldown_s": self.WARM_COOLDOWN_S,
                }
            }
        }
        warm = mock.Mock(return_value=warm_ok)
        if warm_error is not None:
            warm.side_effect = warm_error
        post = mock.Mock(return_value=mock.Mock(status_code=200))
        patches = {
            "_load_server_config_payload": mock.Mock(return_value=payload),
            "load_catalog_with_diagnostics": mock.Mock(return_value=(catalog, None)),
            "cached_model_gpu_sets": mock.Mock(return_value=dict(gpu_sets)),
            "detect_cuda_device_count": mock.Mock(return_value=total_gpus),
            "get_published_model_ids": mock.Mock(return_value=set(published)),
            "get_loaded_model_ids": mock.Mock(return_value=set(loaded)),
            "get_llama_server_processes": mock.Mock(return_value=processes),
            "get_gpu_process_memory_by_pid": mock.Mock(return_value=per_pid),
            "_query_gpu_memory_snapshot_cached": mock.Mock(return_value=snapshot),
"estimate_model_runtime_mib": mock.Mock(
                side_effect=lambda model: (sizes or {}).get(
                    model.model_id, ROUTE_VARIANT_NEEDED_MIB
                )
            ),
            "model_has_enough_free_vram_to_load": mock.Mock(
                return_value=(fit, fit_info or {"reason": "fits", "gpu_set": [0]})
            ),
            "disposable_route_in_flight": mock.Mock(
                side_effect=lambda route_id: int(flights.get(route_id, 0))
            ),
            "log_api_event": mock.Mock(
                side_effect=lambda event, data=None, **_kw: events.append((event, data or {}))
            ),
        }
        if real_http:
            patches["requests"] = mock.Mock(post=post)
        else:
            patches["warm_route_via_llamaswap"] = warm
        with contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(_cli_impl, name, value))
            yield _WarmerProbes(warm=warm, post=post, events=events, patches=patches)

    def _tick(self, offset_s: float = 0.0) -> str:
        return _cli_impl.run_idle_warmer_tick(
            self.args, now=self.WARM_T0 + float(offset_s)
        )

    def test_the_default_config_warms_nothing_and_touches_no_probe(self):
        with self._warmer(published={TARGET_ID}) as probes:
            with mock.patch.object(
                _cli_impl, "_load_server_config_payload", return_value={}
            ):
                self.assertEqual(self._tick(self.WARM_IDLE_AFTER_S + 1), "")
        probes.warm.assert_not_called()
        probes.patches["detect_cuda_device_count"].assert_not_called()

    def test_a_card_idle_past_the_threshold_gets_exactly_one_warm_up(self):
        with self._warmer(published={TARGET_ID}) as probes:
            self._tick(0.0)
            warmed = self._tick(self.WARM_IDLE_AFTER_S + 1)
        self.assertEqual(warmed, TARGET_ID)
        self.assertEqual(
            [call.args for call in probes.warm.call_args_list],
            [(TARGET_ID, "127.0.0.1", 11436)],
            "the warm-up must name the route and the swap endpoint, nothing else",
        )
        self.assertEqual(
            probes.by_kind["idle_warmer_warmed"]["reason"],
            "smallest_fitting",
            "nothing was resident on either card, so the smallest route is the pick",
        )

    def test_a_card_with_a_process_on_it_is_never_warmed(self):
        # Card 0 holds the base route and a process; card 1 is empty and holds the
        # published variant. Warming the variant is the whole point, warming card 0 would
        # evict the process that is already there.
        with self._warmer(
            published={TARGET_ID, self.variant_id}, occupant_cards=(0,)
        ) as probes:
            self._tick(0.0)
            warmed = self._tick(self.WARM_IDLE_AFTER_S + 1)
        self.assertEqual(warmed, self.variant_id)
        self.assertEqual(probes.by_kind["idle_warmer_warmed"]["gpu"], 1)

    def test_a_card_that_is_only_just_idle_is_left_alone(self):
        with self._warmer(published={TARGET_ID}) as probes:
            self._tick(0.0)
            self._tick(self.WARM_IDLE_AFTER_S - 1)
        probes.warm.assert_not_called()
        self.assertNotIn("idle_warmer_warmed", probes.kinds)

    def test_a_resident_copy_outranks_a_smaller_cold_one(self):
        # The rule keys on the cost of the load, never on a model id. ``resident`` is
        # loaded on the busy card, so its bytes are already in the page cache and a second
        # copy on the empty one is a re-map; ``cold`` would be read off storage from
        # scratch even though it is six times smaller.
        cold = _make_model("cold-model", str(self.root / "cold.safetensors"))
        published = {TARGET_ID, self.other_variant_id, "cold-model"}
        with self._warmer(
            catalog=[self.target, cold],
            gpu_sets={TARGET_ID: [1], "cold-model": [0]},
            sizes={TARGET_ID: 30000.0, "cold-model": 5000.0},
            published=published,
            loaded={TARGET_ID},
            occupant_cards=(1,),
        ) as probes:
            ordered = _cli_impl.idle_warm_candidates(
                0, [self.target, cold], 2, published=published, loaded={TARGET_ID}
            )
            self._tick(0.0)
            warmed = self._tick(self.WARM_IDLE_AFTER_S + 1)
        self.assertEqual(
            [entry.route_id for entry in ordered],
            [self.other_variant_id, "cold-model"],
            "the resident tier sorts first no matter how much smaller the cold route is",
        )
        self.assertEqual(warmed, self.other_variant_id)
        self.assertEqual(probes.by_kind["idle_warmer_warmed"]["reason"], "weights_resident")

    def test_the_smallest_route_on_the_card_is_chosen_when_nothing_is_resident(self):
        smaller = _make_model("small-model", str(self.root / "small.safetensors"))
        with self._warmer(
            catalog=[self.target, smaller],
            gpu_sets={TARGET_ID: [0], "small-model": [0]},
            sizes={TARGET_ID: 30000.0, "small-model": 2000.0},
            published={TARGET_ID, "small-model"},
        ) as probes:
            ordered = _cli_impl.idle_warm_candidates(
                0,
                [self.target, smaller],
                2,
                published={TARGET_ID, "small-model"},
                loaded=set(),
            )
            self._tick(0.0)
            warmed = self._tick(self.WARM_IDLE_AFTER_S + 1)
        self.assertEqual([entry.route_id for entry in ordered], ["small-model", TARGET_ID])
        self.assertEqual(warmed, "small-model")
        self.assertEqual(probes.by_kind["idle_warmer_warmed"]["reason"], "smallest_fitting")

    def test_a_route_that_cannot_fit_is_refused_once_and_then_the_card_goes_quiet(self):
        with self._warmer(published={TARGET_ID}, fit=False) as probes:
            self._tick(0.0)
            self.assertEqual(self._tick(self.WARM_IDLE_AFTER_S + 1), "")
            first_pass = list(probes.events)
            self.assertEqual(self._tick(self.WARM_IDLE_AFTER_S + 2), "")
        probes.warm.assert_not_called()
        refusal = probes.by_kind["idle_warmer_refused"]
        self.assertEqual(refusal["reason"], "insufficient_vram")
        self.assertEqual(refusal["route"], TARGET_ID)
        self.assertEqual(refusal["gpu"], 0)
        self.assertEqual(
            probes.events[len(first_pass):],
            [],
            "the cooldown has to silence the card, or the loop re-refuses every tick",
        )

    def test_a_route_that_is_already_loaded_is_never_re_warmed(self):
        with self._warmer(published={TARGET_ID}, loaded={TARGET_ID}) as probes:
            self._tick(0.0)
            self.assertEqual(self._tick(self.WARM_IDLE_AFTER_S + 1), "")
        probes.warm.assert_not_called()
        self.assertEqual(probes.by_kind["idle_warmer_refused"]["reason"], "already_loaded")

    def test_a_model_with_a_request_in_flight_is_never_warmed(self):
        with self._warmer(published={TARGET_ID}, in_flight={TARGET_ID: 1}) as probes:
            self._tick(0.0)
            self.assertEqual(self._tick(self.WARM_IDLE_AFTER_S + 1), "")
        probes.warm.assert_not_called()
        self.assertEqual(probes.by_kind["idle_warmer_refused"]["reason"], "model_in_flight")

    def test_a_variant_reads_its_base_counter_so_a_streaming_base_blocks_the_warm_up(self):
        # disposable_route_in_flight already folds a __gpuN's base counter in; the warmer
        # has to route every guard through it, or a streaming base reads as idle here.
        seen: list[str] = []
        with self._warmer(published={TARGET_ID, self.variant_id}) as probes:
            with mock.patch.object(
                _cli_impl,
                "disposable_route_in_flight",
                side_effect=lambda route_id: seen.append(route_id) or 1,
            ):
                self._tick(0.0)
                self._tick(self.WARM_IDLE_AFTER_S + 1)
        self.assertIn(self.variant_id, seen)
        probes.warm.assert_not_called()
        self.assertEqual(probes.by_kind["idle_warmer_refused"]["reason"], "model_in_flight")

    def test_two_idle_cards_only_load_one_of_them_per_tick(self):
        other = _make_model("second-model", str(self.root / "second.safetensors"))
        with self._warmer(
            catalog=[self.target, other],
            gpu_sets={TARGET_ID: [0], "second-model": [1]},
            published={TARGET_ID, "second-model"},
        ) as probes:
            self._tick(0.0)
            first = self._tick(self.WARM_IDLE_AFTER_S + 1)
        self.assertEqual(first, TARGET_ID)
        self.assertEqual(
            len(probes.warm.call_args_list),
            1,
            "a fleet of idle cards must not become a fleet of simultaneous loads",
        )

    def test_a_warm_up_that_raises_is_contained_and_the_next_tick_still_runs(self):
        with self._warmer(published={TARGET_ID}, warm_error=RuntimeError("swap is gone")) as probes:
            self._tick(0.0)
            self.assertEqual(self._tick(self.WARM_IDLE_AFTER_S + 1), "")
            with mock.patch.object(
                probes.warm, "side_effect", None
            ), mock.patch.object(probes.warm, "return_value", True):
                self.assertEqual(self._tick(self.WARM_IDLE_AFTER_S + 2), TARGET_ID)
        self.assertEqual(probes.by_kind["idle_warmer_failed"]["route"], TARGET_ID)
        self.assertIn("idle_warmer_warmed", probes.kinds)
        self.assertEqual(
            _cli_impl._IDLE_WARMER_STATE.warming,
            {},
            "a failed claim must not stay held after the tick that took it",
        )

    def test_the_daemon_loop_contains_a_tick_that_raises(self):
        # The tick is total on its own; the loop is the second net that keeps a broken
        # probe from taking the manager's warmer thread down with it.
        source = inspect.getsource(daemon_mod.start_idle_warmer)
        self.assertIn("run_idle_warmer_tick", source)
        self.assertIn("idle_warmer_error", source)
        self.assertLess(
            source.index("run_idle_warmer_tick"), source.index("idle_warmer_error")
        )

    def test_the_warm_up_request_is_minimal_and_carries_the_variant_id(self):
        # The base is pinned to the busy card, so the only thing that can land on the
        # empty one is its published __gpuN route. A base id here would load a second copy
        # of the same weights on the card the process is already on.
        with self._warmer(
            published={TARGET_ID, self.other_variant_id},
            gpu_sets={TARGET_ID: [1]},
            occupant_cards=(1,),
            real_http=True,
        ) as probes:
            self._tick(0.0)
            warmed = self._tick(self.WARM_IDLE_AFTER_S + 1)
        self.assertEqual(warmed, self.other_variant_id)
        probes.post.assert_called_once()
        url = probes.post.call_args.args[0]
        self.assertEqual(url, "http://127.0.0.1:11436/v1/chat/completions")
        body = probes.post.call_args.kwargs["json"]
        self.assertEqual(
            body["model"],
            self.other_variant_id,
            "the card that is idle is not the one the base is pinned to, so the warm copy "
            "has to be its __gpuN route",
        )
        self.assertNotEqual(body["model"], TARGET_ID)
        self.assertEqual(body["max_tokens"], 1)
        self.assertFalse(body["stream"])
        self.assertEqual(len(body["messages"][0]["content"]), 1)

    def test_a_card_already_being_warmed_is_not_warmed_again(self):
        with self._warmer(published={TARGET_ID}) as probes:
            _cli_impl._IDLE_WARMER_STATE.warming[0] = TARGET_ID
            try:
                self.assertEqual(self._tick(self.WARM_IDLE_AFTER_S + 1), "")
            finally:
                _cli_impl._IDLE_WARMER_STATE.warming.clear()
        probes.warm.assert_not_called()
        self.assertEqual(probes.by_kind["idle_warmer_refused"]["reason"], "warm_in_flight")

    def test_a_route_being_warmed_elsewhere_is_never_warmed_twice(self):
        with self._warmer(published={TARGET_ID, self.variant_id}) as probes:
            _cli_impl._IDLE_WARMER_STATE.warming[1] = TARGET_ID
            try:
                self.assertEqual(self._tick(self.WARM_IDLE_AFTER_S + 1), "")
            finally:
                _cli_impl._IDLE_WARMER_STATE.warming.clear()
        probes.warm.assert_not_called()
        self.assertEqual(probes.by_kind["idle_warmer_refused"]["reason"], "warm_in_flight")

    def test_an_occupied_card_forgets_its_idle_clock(self):
        with self._warmer(published={TARGET_ID}, occupant_cards=(0,)) as probes:
            self._tick(0.0)
            probes.warm.assert_not_called()
        self.assertNotIn(
            0,
            _cli_impl._IDLE_WARMER_STATE.idle_since,
            "the clock may only run while a card is empty, so a process on it resets it",
        )


class IdleWarmerWiringTest(unittest.TestCase):
    """The warmer is a loop over requests. Nothing in it may reach a config writer."""

    @staticmethod
    def _warmer_source() -> str:
        names = (
            "_resolve_idle_gpu_warmer_config",
            "_idle_warmer_occupied_cards",
            "idle_warm_candidates",
            "warm_route_via_llamaswap",
            "run_idle_warmer_tick",
            "_warm_idle_card",
        )
        return "\n".join(inspect.getsource(getattr(_cli_impl, name)) for name in names) + inspect.getsource(
            daemon_mod.start_idle_warmer
        )

    def test_the_warmer_reaches_no_config_writer(self):
        # llama-swap runs --watch-config, so ANY config.yaml write tears down every loaded
        # model: a warmer that rendered the config would destroy the fleet it maintains.
        source = self._warmer_source()
        for name in (
            "render_llamaswap_config",
            "_write_config_atomic",
            "ensure_internal_route",
            "ensure_replica_route",
            "drop_internal_route",
            "set_instance_mmproj",
            "save_catalog",
            "subprocess",
            "Popen",
        ):
            self.assertNotIn(name, source, name)

    def test_the_only_load_mechanism_is_the_swap_request(self):
        source = inspect.getsource(_cli_impl.warm_route_via_llamaswap)
        self.assertIn("/v1/chat/completions", source)
        self.assertIn('"max_tokens": 1', source)
        self.assertNotIn("subprocess", source)

    def test_the_warmer_switch_stays_out_of_the_migrated_experimental_block(self):
        # A default-filled key under `experimental` would come back out of
        # config-migrate, which is read as "a route needs to change".
        self.assertNotIn("idle_gpu_warmer", _cli_impl._default_experimental_config())
        self.assertIn("idle_gpu_warmer", _cli_impl._normalize_experimental_config({"idle_gpu_warmer": {"enabled": True}}))


class StaticReplicaRouteTest(unittest.TestCase):
    """The render publishes replica routes, so no request writes config.yaml.

    llama-swap runs ``--watch-config``: any write to config.yaml unloads every
    loaded model and destroys every KV cache. Replicas used to reach llama-swap
    only through ``ensure_internal_route_in_llamaswap_config``, so every scale-out
    paid a fleet-wide teardown. They are rendered here instead, the same treatment
    the per-card ``__gpuN`` variants already get.
    """

    HEAVY_A_MIB = 16000
    HEAVY_B_MIB = 14650
    FITS_MIB = 5000
    #: The live catalog's refused shape: one footprint is already over half a card.
    DOES_NOT_FIT_MIB = 26704
    #: Two of these weights plus the cushion fit the card, two of their runtime
    #: footprints do not - so only this model can tell a weakened capacity rule from
    #: the one that measures the process.
    WEIGHTS_FIT_RUNTIME_DOES_NOT_MIB = 10000

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # The assignment memo is keyed on catalog identity and every test here builds
        # a different catalog under the same model ids, so it must not outlive one.
        self._forget_assignment()

    def _forget_assignment(self) -> None:
        R._MODEL_GPU_ASSIGNMENT_MEMO["key"] = None
        self.addCleanup(lambda: R._MODEL_GPU_ASSIGNMENT_MEMO.__setitem__("key", None))

    def _weights(self, model_id: str, mib: int, **fields) -> R.ManagedModel:
        path = self.root / f"{model_id}.safetensors"
        with open(path, "wb") as handle:
            handle.truncate(mib * 1024 * 1024)
        return replace(_make_model(model_id, str(path)), **fields)

    def _lazy_vision(self, model_id: str, mib: int) -> R.ManagedModel:
        """A lazy projector model: the render publishes it text-only, which is the
        state ``set_instance_mmproj_in_llamaswap_config`` later has to change.

        It carries an alias, so a replica that inherited one would show up as a second
        public name for the same weights.
        """
        return self._weights(
            model_id,
            mib,
            mmproj_path=str(self.root / f"{model_id}-mmproj.gguf"),
            mmproj_filename=f"{model_id}-mmproj.gguf",
            aliases=[f"{model_id}-alias"],
            server_overrides={"mmproj_mode": "lazy"},
        )

    @staticmethod
    def _cfg(**overrides) -> dict[str, object]:
        return {
            "enabled": True,
            "max": 1,
            "placement": "exclusive_gpus",
            "gpu_capacity_mib": CARD_CAPACITY_MIB,
            "safety_vram_mib": SHARING_SAFETY_MIB,
        } | overrides

    def _catalog(self) -> list[R.ManagedModel]:
        # heavy-a and heavy-b own both cards, so the only card the third model can
        # put a second copy on is the one its own base is pinned to.
        return [
            self._weights("heavy-a", self.HEAVY_A_MIB),
            self._weights("heavy-b", self.HEAVY_B_MIB),
            self._lazy_vision("shares", self.FITS_MIB),
        ]

    def _render(self, path, cfg=None, catalog=None) -> dict:
        catalog = self._catalog() if catalog is None else catalog
        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            R.render_llamaswap_config(
                catalog, path, SERVER, 11436, 300, {},
                self._cfg() if cfg is None else cfg,
            )
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    @staticmethod
    def _cuda_visible_devices(entry: dict) -> list[int]:
        cmd = str(entry.get("cmd") or "")
        match = re.search(r"CUDA_VISIBLE_DEVICES=([0-9,]+)", cmd)
        assert match is not None, cmd
        return [int(part) for part in match.group(1).split(",") if part.strip()]

    def _runtime_replica_sets(self, catalog) -> dict[str, list[int]]:
        """What the request path computes, called exactly as it calls it."""
        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            return {
                model.model_id: list(gpu_set or [])
                for model, public_base, gpu_set in R.iter_catalog_with_replicas(
                    catalog, self._cfg()
                )
                if public_base is not None
            }

    def _matrix_peers(self, matrix: dict, model_id: str) -> frozenset[str]:
        """The routes this one may be loaded next to.

        Matrix set names are positional, so the group a route lands in is compared as
        the set of its peers rather than by name.
        """
        model_of_var = dict(matrix.get("vars") or {})
        mine = {var for var, mid in model_of_var.items() if mid == model_id}
        peers: set[str] = set()
        for expression in (matrix.get("sets") or {}).values():
            group = {var for var in model_of_var if var in str(expression).split(" & ")}
            if group & mine:
                peers |= {model_of_var[var] for var in group}
        return frozenset(peers - {model_id})

    def test_the_render_publishes_a_replica_route_for_an_admitted_placement(self):
        catalog = self._catalog()
        self.assertEqual(
            set(self._runtime_replica_sets(catalog)),
            {"heavy-a__replica_0", "heavy-b__replica_0", "shares__replica_0"},
            "precondition: these are exactly the admitted placements",
        )
        rendered = self._render(self.root / "config.yaml", catalog=catalog)
        for replica_id in ("heavy-a__replica_0", "heavy-b__replica_0", "shares__replica_0"):
            entry = rendered["models"][replica_id]
            self.assertEqual(
                entry["metadata"],
                {"internal_replica_of": replica_id.removesuffix("__replica_0")},
            )
            self.assertEqual(entry["checkEndpoint"], "/health")
            self.assertEqual(entry["ttl"], 300)
            self.assertIn("capabilities", entry)

    def test_the_replica_route_carries_its_own_gpu_set_in_the_command_and_the_matrix(self):
        catalog = self._catalog()
        expected = self._runtime_replica_sets(catalog)["shares__replica_0"]
        rendered = self._render(self.root / "config.yaml", catalog=catalog)
        entry = rendered["models"]["shares__replica_0"]
        cmd = str(entry["cmd"])
        self.assertEqual(cmd.count("CUDA_VISIBLE_DEVICES="), 1, cmd)
        self.assertEqual(self._cuda_visible_devices(entry), expected)
        # The matrix rebuilds an entry's gpu_set out of its own command, so this is
        # the gpu_set the rendered replica was placed with, not the base's.
        self.assertEqual(
            R._model_info_for_matrix_entry(
                "shares__replica_0", entry, {"shares": catalog[2]}
            )["gpu_set"],
            expected,
        )
        matrix = rendered["matrix"]
        var_of = {mid: var for var, mid in (matrix["vars"] or {}).items()}
        self.assertEqual(
            matrix["evict_costs"][var_of["shares__replica_0"]],
            R.route_evict_cost(R._get_model_size_mib(catalog[2]), "shares__replica_0"),
        )
        self.assertIn("shares", self._matrix_peers(matrix, "shares__replica_0"))

    def test_a_replica_route_is_never_published_as_a_public_model(self):
        catalog = self._catalog()
        rendered = self._render(self.root / "config.yaml", catalog=catalog)
        entry = rendered["models"]["shares__replica_0"]
        self.assertNotIn("aliases", entry)
        self.assertEqual(rendered["models"]["shares"]["aliases"], ["shares-alias"])
        self.assertEqual(catalog[2].aliases, ["shares-alias"])
        # The public list is the gateway's own, built from the catalog and never from
        # config.yaml, so a replica route cannot reach a client even by id.
        public = {
            _cli_impl.build_openai_model_list_payload(model)["id"] for model in catalog
        }
        self.assertEqual(public, {"heavy-a", "heavy-b", "shares"})
        self.assertFalse([mid for mid in public if R.is_replica_model_id(mid)])

    def test_base_routes_are_byte_identical_to_the_bases_only_render(self):
        catalog = self._catalog()
        before = self._render(self.root / "before.yaml", self._cfg(enabled=False), catalog)
        after = self._render(self.root / "after.yaml", self._cfg(), catalog)
        self.assertEqual(
            set(after["models"]) - set(before["models"]),
            {"heavy-a__replica_0", "heavy-b__replica_0", "shares__replica_0"},
        )
        self.assertEqual(set(before["models"]) - set(after["models"]), set())
        for model_id in sorted(before["models"]):
            # Nothing about a base route may move: the replica tiers only ever add
            # routes, and the command each base launches with is not one of them.
            self.assertEqual(after["models"][model_id], before["models"][model_id], model_id)
            self.assertEqual(
                self._cost(after["matrix"], model_id),
                self._cost(before["matrix"], model_id),
                model_id,
            )
            # A new route on a card of its own may only ever add a peer: a route
            # sitting on another card becomes co-loadable with it. A pair that used to
            # be declared together must never stop being declared.
            self.assertLessEqual(
                self._matrix_peers(before["matrix"], model_id),
                self._matrix_peers(after["matrix"], model_id),
                model_id,
            )
        for base in ("shares", "heavy-a", "heavy-b"):
            self.assertIn(base, self._matrix_peers(after["matrix"], f"{base}__replica_0"))

    @staticmethod
    def _cost(matrix: dict, model_id: str) -> int | None:
        var_of = {mid: var for var, mid in (matrix.get("vars") or {}).items()}
        return (matrix.get("evict_costs") or {}).get(var_of.get(model_id))

    def test_a_model_whose_copies_do_not_fit_publishes_no_replica_route(self):
        refused = ("qwen3.6-27b-nvfp4", self.DOES_NOT_FIT_MIB)
        kv_hungry = ("kv-hungry", self.WEIGHTS_FIT_RUNTIME_DOES_NOT_MIB)
        catalog = [
            self._weights(*refused),
            self._weights(*kv_hungry),
            self._weights("fits", self.FITS_MIB),
        ]
        runtime = models_mod.estimate_model_runtime_mib(catalog[0])
        cfg = R.ReplicaConfig(
            enabled=True, max=1, placement="exclusive_gpus",
            gpu_capacity_mib=CARD_CAPACITY_MIB, safety_vram_mib=SHARING_SAFETY_MIB,
        )
        self.assertGreater(float(runtime), CARD_CAPACITY_MIB / 2)
        self.assertGreater(2 * float(runtime) + SHARING_SAFETY_MIB, CARD_CAPACITY_MIB)
        self.assertFalse(R._replica_fits_cards(catalog[0], cfg, [0], {0}))
        # The second model's weights alone would fit the card; only its whole-process
        # footprint does not, so a capacity rule weakened to compare weights admits it.
        self.assertLessEqual(
            2 * R._get_model_size_mib(catalog[1]) + SHARING_SAFETY_MIB, CARD_CAPACITY_MIB
        )
        self.assertFalse(R._replica_fits_cards(catalog[1], cfg, [1], {1}))
        sets = self._runtime_replica_sets(catalog)
        # One copy of it does fit a card, so the two-copy refusal only closes its own
        # base's card; the replica is published on the card its base is not on.
        self.assertEqual(sets.get(f"{kv_hungry[0]}__replica_0"), [0])
        self.assertNotIn(f"{refused[0]}__replica_0", sets)
        self.assertEqual(set(sets), {f"{kv_hungry[0]}__replica_0", "fits__replica_0"})
        rendered = self._render(self.root / "config.yaml", self._cfg(), catalog)
        self.assertNotIn(f"{refused[0]}__replica_0", rendered["models"])
        self.assertIn(refused[0], rendered["models"])
        for model_id, gpu_set in sets.items():
            self.assertIn(model_id, rendered["models"])
            self.assertEqual(self._cuda_visible_devices(rendered["models"][model_id]), gpu_set)

    def test_a_base_owned_neighbours_card_publishes_the_replica_route_there(self):
        # The EXL3 shape end to end: a big model whose base owns card 0, on a host
        # where every card is some base, so its only replica is on card 1 - the card
        # its base does *not* own. This is the whole point of the tier: two copies,
        # one per card, instead of no replica at all.
        catalog = [
            self._weights("qwen3.8-27b-EXL3", 18155),
            self._weights("other-base", self.HEAVY_B_MIB),
        ]
        runtime_mib = float(models_mod.estimate_model_runtime_mib(catalog[0]))
        self.assertLessEqual(runtime_mib + SHARING_SAFETY_MIB, CARD_CAPACITY_MIB)
        self.assertGreater(2 * runtime_mib + SHARING_SAFETY_MIB, CARD_CAPACITY_MIB)

        sets = self._runtime_replica_sets(catalog)
        replica = sets["qwen3.8-27b-EXL3__replica_0"]
        self.assertEqual(replica, [1])
        self.assertNotIn(0, replica)
        self.assertEqual(R.assign_model_gpu_sets(catalog, 2)["qwen3.8-27b-EXL3"], [0])
        rendered = self._render(self.root / "config.yaml", self._cfg(), catalog)
        entry = rendered["models"]["qwen3.8-27b-EXL3__replica_0"]
        self.assertEqual(self._cuda_visible_devices(entry), [1])
        self.assertEqual(self._cuda_visible_devices(rendered["models"]["qwen3.8-27b-EXL3"]), [0])

    def test_a_model_too_big_for_one_card_publishes_no_replica_route_at_any_tier(self):
        catalog = [
            self._weights("qwen3.6-27b-nvfp4", self.DOES_NOT_FIT_MIB),
            self._weights("neighbour", self.HEAVY_B_MIB),
        ]
        model = catalog[0]
        runtime_mib = float(models_mod.estimate_model_runtime_mib(model))
        cfg = R.ReplicaConfig(
            enabled=True, max=1, placement="exclusive_gpus",
            gpu_capacity_mib=CARD_CAPACITY_MIB, safety_vram_mib=SHARING_SAFETY_MIB,
        )
        # Its own share alone does not fit a card, so neither the two-copy rule on
        # its own base's card nor the single-share rule on a neighbour's can admit it.
        self.assertGreater(runtime_mib + SHARING_SAFETY_MIB, CARD_CAPACITY_MIB)
        self.assertFalse(R._replica_fits_cards(model, cfg, [0], {0}))
        self.assertFalse(R._replica_fits_cards(model, cfg, [1], {0}))

        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            sets = self._runtime_replica_sets(catalog)
            lines = R.summarize_configured_replicas(catalog, self._cfg())
        self.assertNotIn("qwen3.6-27b-nvfp4__replica_0", sets)
        refusal = next(line for line in lines if line.startswith("qwen3.6-27b-nvfp4:"))
        self.assertIn("replicas enabled but no replica card", refusal)

        rendered = self._render(self.root / "config.yaml", self._cfg(), catalog)
        self.assertNotIn("qwen3.6-27b-nvfp4__replica_0", rendered["models"])
        self.assertIn("qwen3.6-27b-nvfp4", rendered["models"])

    def test_rendering_twice_over_an_unchanged_catalog_produces_the_same_file(self):
        catalog = self._catalog()
        first = self.root / "first.yaml"
        second = self.root / "second.yaml"
        self._render(first, catalog=catalog)
        self._forget_assignment()
        with _no_live_gpu_state() as probes:
            self._render(second, catalog=catalog)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        for probe in probes.values():
            probe.assert_not_called()

    def test_ordinary_text_traffic_writes_no_config(self):
        catalog = self._catalog()
        path = self.root / "config.yaml"
        self._render(path, catalog=catalog)
        before = path.read_bytes()
        before_mtime = path.stat().st_mtime_ns
        gpu_set = self._runtime_replica_sets(catalog)["shares__replica_0"]
        with _no_live_gpu_state() as probes, mock.patch.object(
            R, "detect_cuda_device_count", return_value=2
        ):
            R.ensure_replica_route_in_llamaswap_config(
                catalog[2], 0, gpu_set, catalog, path, SERVER, 300, {},
            )
        self.assertEqual(path.read_bytes(), before, "the render already published it")
        self.assertEqual(path.stat().st_mtime_ns, before_mtime)
        for probe in probes.values():
            probe.assert_not_called()

    def test_the_vision_attach_still_writes_exactly_one_route(self):
        catalog = self._catalog()
        path = self.root / "config.yaml"
        rendered = self._render(path, catalog=catalog)
        before = rendered["models"]
        self.assertNotIn(
            "--mmproj", before["shares__replica_0"]["cmd"], "precondition: lazy"
        )
        with mock.patch.object(R, "detect_cuda_device_count", return_value=2):
            R.set_instance_mmproj_in_llamaswap_config(
                catalog[2], "shares__replica_0", catalog, path, SERVER, 300, {},
                enable=True, gpu_set=self._cuda_visible_devices(before["shares__replica_0"]),
            )
        after = yaml.safe_load(path.read_text(encoding="utf-8"))["models"]
        self.assertIn("--mmproj", after["shares__replica_0"]["cmd"])
        changed = [mid for mid in sorted(after) if after[mid] != before.get(mid)]
        self.assertEqual(changed, ["shares__replica_0"])


if __name__ == "__main__":
    unittest.main()