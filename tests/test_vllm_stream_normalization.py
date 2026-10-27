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


def test_relay_sse_chunks_reassembles_lines_split_across_chunks():
    from llamacpp_stack._cli_impl import _relay_sse_chunks

    chunk = {"choices": [{"index": 0, "delta": {"reasoning": "a", "content": "b"}, "finish_reason": None}]}
    frame = b"data: " + json.dumps(chunk).encode() + b"\n\n"
    out = []
    _relay_sse_chunks([frame[:9], frame[9:31], frame[31:]], out.append)
    frames = [f for f in b"".join(out).split(b"\n\n") if f]
    assert len(frames) == 2
    for raw in frames:
        delta = json.loads(raw[6:])["choices"][0]["delta"]
        assert not (delta.get("content") and delta.get("reasoning_content"))


def test_relay_sse_chunks_preserves_multibyte_split():
    from llamacpp_stack._cli_impl import _relay_sse_chunks

    payload = b"data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "\u00f1"}, "finish_reason": None}]}, ensure_ascii=False).encode() + b"\n\n"
    cut = payload.index(b"\xc3") + 1
    out = []
    _relay_sse_chunks([payload[:cut], payload[cut:]], out.append)
    assert b"".join(out) == payload


def test_normalize_json_reasoning_body_renames_message_field():
    from llamacpp_stack._cli_impl import _normalize_openai_json_reasoning_body

    body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "x", "reasoning": "think"}}]}).encode()
    message = json.loads(_normalize_openai_json_reasoning_body(body))["choices"][0]["message"]
    assert message["reasoning_content"] == "think"
    assert "reasoning" not in message
