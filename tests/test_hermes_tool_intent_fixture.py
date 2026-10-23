import json
import unittest
from pathlib import Path

from llamacpp_stack.cli import _chat_tool_continue_trigger_reason, _normalize_openai_message

FIXTURES = Path(__file__).parent / "fixtures" / "hermes_tool_intent"
DEFAULT_PREFIXES = [
    "[terminal command",
    "[terminal_inline",
    "</terminal_inline>",
    "Voy a",
    "Empezando por",
]

if not FIXTURES.exists():
    raise unittest.SkipTest("hermes_tool_intent fixtures are local-only (gitignored)")


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_captured_request_ends_with_failed_intent_turn():
    req = _load("request-378msgs.json")
    assert req["model"] == "qwen3.8-27b-EXL3"
    assert len(req["messages"]) == 378
    assert len(req["tools"]) == 37

    failed = req["messages"][-2]
    assert failed["role"] == "assistant"
    assert str(failed["content"]).startswith("Ejecuto el probe definitivo del picker de carpeta:")
    assert not failed.get("tool_calls")

    assert req["messages"][-1]["role"] == "user"
    assert str(req["messages"][-1]["content"]).startswith("continua")


def test_repro_pre_failure_drops_failed_turn_and_continua():
    upstream = _load("upstream-378msgs.json")
    repro = _load("repro-pre-failure.json")
    assert len(repro["messages"]) == len(upstream["messages"]) - 2
    assert repro["messages"][-1]["role"] == "user"
    assert "Ejecuto el probe definitivo" not in str(repro["messages"][-1].get("content"))
    assert repro["messages"] == upstream["messages"][:-2]


def test_trigger_fires_on_captured_intent_turn():
    req = _load("request-378msgs.json")
    failed_content = str(req["messages"][-2]["content"])
    reason = _chat_tool_continue_trigger_reason(failed_content, [], req["tools"], DEFAULT_PREFIXES)
    assert reason == "visible_content_trailing_colon"


def test_intent_only_examples_are_present_and_picker_cases_fire():
    req = _load("request-378msgs.json")
    examples = _load("intent_only_examples.json")

    intent_like = [e for e in examples if e.get("intent_only_like")]
    assert len(intent_like) >= 10

    picker = [e for e in examples if "picker de carpetas" in (e.get("visible_preview") or "")]
    assert picker
    for example in picker:
        reason = _chat_tool_continue_trigger_reason(
            example["visible_preview"], [], req["tools"], DEFAULT_PREFIXES
        )
        assert reason, example


def test_normalization_preserves_tool_call_protocol():
    req = _load("request-378msgs.json")
    raw = [m for m in req["messages"] if isinstance(m, dict)]
    normalized = [_normalize_openai_message(m) for m in raw]

    assert sum(1 for m in normalized if m.get("tool_calls")) == 175
    assert sum(1 for m in normalized if m.get("role") == "tool") == 185
    assert sum(1 for m in normalized if m.get("tool_call_id")) == 185

    assistant_call = normalized[-4]
    tool_result = normalized[-3]
    assert assistant_call["role"] == "assistant"
    assert tool_result["role"] == "tool"
    assert assistant_call["tool_calls"][0]["id"] == tool_result["tool_call_id"]
