"""Tests del monitor de calidad de respuestas."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llamacpp_stack.cli import response_quality as rq

SPANISH_PROMPT = "Revisa esta captura y dime si el hero movil se ve correcto y profesional. Responde si o no."


class ConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = rq.normalize_config(None)
        self.assertTrue(cfg["enabled"])
        self.assertTrue(cfg["save_cases"])

    def test_bounds_and_idempotent(self) -> None:
        cfg = rq.normalize_config({"language_drift_ratio": 5, "fragmentation_min_lines": 1})
        self.assertEqual(cfg["language_drift_ratio"], 1.0)
        self.assertEqual(cfg["fragmentation_min_lines"], 5)
        self.assertEqual(rq.normalize_config(cfg), cfg)


class AnalyzeTests(unittest.TestCase):
    def _a(self, content, reasoning="", finish="stop", tool_calls=0, prompt=SPANISH_PROMPT, chunks=None):
        return rq.analyze(prompt_text=prompt, content=content, reasoning=reasoning,
                          finish_reason=finish, tool_calls=tool_calls, chunk_sizes=chunks)

    def test_clean_response_has_no_flags(self) -> None:
        m = self._a("El hero movil se ve correcto: titular legible, CTA visible y contraste suficiente.")
        self.assertEqual(m["flags"], [])

    def test_empty(self) -> None:
        self.assertIn("empty", self._a("")["flags"])

    def test_empty_with_tool_calls_is_not_flagged(self) -> None:
        self.assertEqual(self._a("", finish="tool_calls", tool_calls=3)["flags"], [])

    def test_reasoning_only(self) -> None:
        flags = self._a("", reasoning="pensando sobre el layout")["flags"]
        self.assertIn("reasoning_only", flags)
        self.assertIn("empty", flags)

    def test_truncated(self) -> None:
        self.assertIn("truncated", self._a("Texto que se corta", finish="length")["flags"])

    def test_language_drift(self) -> None:
        content = "收到你的怀疑是对的，所以我把验证做成你能直接看到的：以后每轮晋升都会附上完整滚动。" * 2
        m = self._a(content)
        self.assertIn("language_drift", m["flags"])
        self.assertGreater(m["cjk_ratio"], 0.5)

    def test_chinese_prompt_is_not_drift(self) -> None:
        content = "收到你的怀疑是对的，所以我把验证做成你能直接看到的完整滚动。" * 2
        prompt = "请检查这张截图并告诉我英雄区是否正常。" * 3
        m = self._a(content, prompt=prompt)
        self.assertNotIn("language_drift", m["flags"])

    def test_fragmentation(self) -> None:
        content = "\n".join(["de", "la", "se", "no", "de"] * 12)
        m = self._a(content)
        self.assertIn("fragmentation", m["flags"])
        self.assertGreaterEqual(m["short_line_ratio"], 0.5)

    def test_normal_multiline_is_not_fragmentation(self) -> None:
        content = "\n".join([f"Linea {i} con contenido suficiente para no ser un fragmento." for i in range(40)])
        self.assertNotIn("fragmentation", self._a(content)["flags"])

    def test_chunk_stats_recorded(self) -> None:
        m = self._a("una respuesta normal con contenido", chunks=[1, 2, 30, 40, 3])
        self.assertEqual(m["chunks"], 5)
        self.assertAlmostEqual(m["tiny_chunk_ratio"], 0.6, places=3)


class SaveCaseTests(unittest.TestCase):
    def test_saves_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = rq.normalize_config({"dir": tmp})
            path = rq.save_case(cfg=cfg, payload={"request_id": "chat_req_test", "flags": ["empty"]},
                                prompt_text="hola", content="", reasoning="x")
            self.assertIsNotNone(path)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(data["request_id"], "chat_req_test")
            self.assertEqual(data["flags"], ["empty"])

    def test_save_disabled(self) -> None:
        cfg = rq.normalize_config({"save_cases": False})
        self.assertIsNone(rq.save_case(cfg=cfg, payload={"request_id": "x"}, prompt_text="", content="", reasoning=""))


if __name__ == "__main__":
    unittest.main()
