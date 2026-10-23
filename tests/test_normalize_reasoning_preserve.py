import llamacpp_stack._cli_impl as impl
from llamacpp_stack.cli import (
    _normalize_openai_message,
    _preserve_thinking_requested,
    _model_preserve_thinking_enabled,
)


def test_reasoning_content_dropped_by_default():
    msg = {"role": "assistant", "content": "hi", "reasoning_content": "why"}
    assert "reasoning_content" not in _normalize_openai_message(msg)


def test_reasoning_content_preserved_when_requested():
    msg = {"role": "assistant", "content": "hi", "reasoning_content": "why"}
    out = _normalize_openai_message(msg, preserve_reasoning_content=True)
    assert out["reasoning_content"] == "why"


def test_reasoning_field_mapped_to_reasoning_content():
    msg = {"role": "assistant", "content": "hi", "reasoning": "vllm style"}
    out = _normalize_openai_message(msg, preserve_reasoning_content=True)
    assert out["reasoning_content"] == "vllm style"


def test_reasoning_content_wins_over_reasoning():
    msg = {"role": "assistant", "content": "hi", "reasoning_content": "a", "reasoning": "b"}
    out = _normalize_openai_message(msg, preserve_reasoning_content=True)
    assert out["reasoning_content"] == "a"


def test_blank_reasoning_not_added():
    for value in ("", "   ", None):
        msg = {"role": "assistant", "content": "hi", "reasoning_content": value, "reasoning": value}
        out = _normalize_openai_message(msg, preserve_reasoning_content=True)
        assert "reasoning_content" not in out


def test_preserve_thinking_requested_parses_variants():
    assert _preserve_thinking_requested('{"preserve_thinking": true}') is True
    assert _preserve_thinking_requested('{"preserve_thinking": "true"}') is True
    assert _preserve_thinking_requested({"preserve_thinking": True}) is True
    assert _preserve_thinking_requested('{"preserve_thinking": false}') is False
    assert _preserve_thinking_requested("{}") is False
    assert _preserve_thinking_requested(None) is False
    assert _preserve_thinking_requested("not json") is False
    assert _preserve_thinking_requested('["preserve_thinking"]') is False


def _stub_model_backend(monkeypatch, backend, ctk):
    monkeypatch.setattr(impl, "_model_backend", lambda model: backend)
    monkeypatch.setattr(impl, "resolve_llama_server_defaults", lambda args: {})
    monkeypatch.setattr(impl, "_effective_llama_server_options", lambda model, defaults: {"chat_template_kwargs": ctk})


def test_model_preserve_thinking_enabled_true(monkeypatch):
    _stub_model_backend(monkeypatch, "llama.cpp", '{"preserve_thinking": true}')
    assert _model_preserve_thinking_enabled(object(), None) is True


def test_model_preserve_thinking_disabled_value(monkeypatch):
    _stub_model_backend(monkeypatch, "llama.cpp", '{"preserve_thinking": false}')
    assert _model_preserve_thinking_enabled(object(), None) is False


def test_model_preserve_thinking_missing_key(monkeypatch):
    _stub_model_backend(monkeypatch, "llama.cpp", '{"enable_thinking": true}')
    assert _model_preserve_thinking_enabled(object(), None) is False


def test_model_preserve_thinking_non_llamacpp(monkeypatch):
    _stub_model_backend(monkeypatch, "vllm", '{"preserve_thinking": true}')
    assert _model_preserve_thinking_enabled(object(), None) is False


def test_model_preserve_thinking_none_entry(monkeypatch):
    _stub_model_backend(monkeypatch, "llama.cpp", '{"preserve_thinking": true}')
    assert _model_preserve_thinking_enabled(None, None) is False
