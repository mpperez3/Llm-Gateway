"""Base-process attribution when a base and its replicas share one weights file.

The base model and every replica launch the same ``--model`` path, so keying
running processes by ``model_path`` attributed whichever process came last in
``ps`` to the base. During a cold replica load the base's runtime snapshot
reported the replica's pid, port and 386 MiB footprint. These tests pin route
resolution to the port llama-swap assigns to each rendered route.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from llamacpp_stack import _cli_impl

BASE_ID = "base-model"
REPLICA_ID = f"{BASE_ID}__replica_0"
VARIANT_ID = f"{BASE_ID}__gpu0"


class ProcessAttributionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model_file = self.root / "model.gguf"
        self.model_file.write_bytes(b"x")
        self.model = SimpleNamespace(
            model_id=BASE_ID,
            local_path=str(self.model_file),
        )
        self.config = self.root / "config.yaml"
        self.config.write_text(
            yaml.safe_dump(
                {
                    "startPort": 18000,
                    "models": {
                        BASE_ID: {"cmd": "base", "metadata": {}},
                        VARIANT_ID: {"cmd": "variant", "metadata": {"route_variant_of": BASE_ID}},
                        REPLICA_ID: {"cmd": "replica", "metadata": {"internal_replica_of": BASE_ID}},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.model_path = _cli_impl._safe_realpath(str(self.model_file))

    def _processes(self, *ports: int) -> list[dict]:
        return [
            {"pid": 1000 + port, "port": port, "model_path": self.model_path, "cmdline": ""}
            for port in ports
        ]

    def _resolve(self, ports: list[int], **kwargs):
        processes = self._processes(*ports)
        return _cli_impl.get_catalog_model_process(
            BASE_ID,
            [self.model],
            config_path=self.config,
            processes=processes,
            **kwargs,
        )

    def test_the_route_index_maps_each_route_to_its_assigned_port(self):
        routes = _cli_impl._llamaswap_route_index(self.config)
        self.assertEqual(routes[BASE_ID]["port"], 18000)
        self.assertEqual(routes[VARIANT_ID]["port"], 18001)
        self.assertEqual(routes[REPLICA_ID]["port"], 18002)
        self.assertTrue(routes[REPLICA_ID]["is_replica"])
        self.assertEqual(routes[VARIANT_ID]["base_model"], BASE_ID)

    def test_the_base_wins_over_a_replica_that_shares_the_weights_file(self):
        process = self._resolve([18002, 18000])
        self.assertEqual(process["port"], 18000)

    def test_the_base_route_is_not_reported_loaded_when_only_its_replica_runs(self):
        self.assertIsNone(self._resolve([18002]))

    def test_a_gpu_variant_counts_as_the_base(self):
        process = self._resolve([18001])
        self.assertEqual(process["port"], 18001)

    def test_display_may_fall_back_to_a_replica_when_the_base_is_absent(self):
        process = self._resolve([18002], allow_replica_fallback=True)
        self.assertEqual(process["port"], 18002)

    def test_without_a_rendered_route_the_path_fallback_is_preserved(self):
        with mock.patch.object(_cli_impl, "get_llama_server_processes", return_value=self._processes(18002)):
            process = _cli_impl.get_catalog_model_process(
                BASE_ID,
                [self.model],
                config_path=self.root / "missing.yaml",
            )
        self.assertEqual(process["port"], 18002)


if __name__ == "__main__":
    unittest.main()
