import json

from llamacpp_stack._cli_impl import _normalize_openai_stream_reasoning_line


def test_stream_normalizer_splits_mixed_reasoning_and_content_frames():
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "model": "m",
        "choices": [{"index": 0, "delta": {"reasoning": " read it.\n", "content": "\n\nYou"}, "finish_reason": None}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    out = _normalize_openai_stream_reasoning_line(b"data: " + json.dumps(chunk).encode())
    frames = [f for f in out.split(b"\n\n") if f]
    assert len(frames) == 2
    first = json.loads(frames[0][6:])
    second = json.loads(frames[1][6:])
    assert first["choices"][0]["delta"] == {"reasoning_content": " read it.\n"}
    assert "usage" not in first
    assert second["choices"][0]["delta"] == {"content": "\n\nYou"}
    assert second["usage"]["completion_tokens"] == 3


def test_stream_normalizer_leaves_single_part_frame_untouched():
    line = b"data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "hola"}, "finish_reason": None}]}).encode()
    assert _normalize_openai_stream_reasoning_line(line) is line


def test_stream_normalizer_keeps_role_on_first_split_frame():
    chunk = {"choices": [{"index": 0, "delta": {"role": "assistant", "reasoning": "a", "content": "b"}, "finish_reason": None}]}
    out = _normalize_openai_stream_reasoning_line(b"data: " + json.dumps(chunk).encode())
    first = json.loads(out.split(b"\n\n")[0][6:])
    assert first["choices"][0]["delta"] == {"reasoning_content": "a", "role": "assistant"}


def test_stream_normalizer_does_not_touch_done_sentinel():
    assert _normalize_openai_stream_reasoning_line(b"data: [DONE]") == b"data: [DONE]"
