#!/usr/bin/env python3
"""Compare two consecutive prompt captures to explain KV-cache reuse loss.

Reads two `tool_intent_capture` JSON files (the gateway's full upstream
payloads) and reports:

  * message-level structure diff (count, roles, first divergent message),
  * character-level common prefix over a canonical rendering of the messages,
  * optionally, the EXACT token LCP by replaying the messages through the
    running llama-server `/apply-template` + `/tokenize` endpoints.

The token LCP matches the server's `n_content_lcp` in the
`edit/divergence sample (... lcp ...)` log line.

Usage:
  # message/char diff only
  tools/compare_prompt_captures.py A.json B.json

  # exact token LCP (model must be loaded)
  tools/compare_prompt_captures.py A.json B.json \
      --base-url http://127.0.0.1:18110 --tokenize

  # pick the two newest captures from a directory
  tools/compare_prompt_captures.py --dir ~/.local/state/llm-server/prompt-captures
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "payload" not in data:
        raise SystemExit(f"{path}: not a tool_intent_capture file")
    return data


def _two_newest(directory: str) -> tuple[str, str]:
    files = sorted(
        glob.glob(os.path.join(os.path.expanduser(directory), "*.json")),
        key=os.path.getmtime,
    )
    if len(files) < 2:
        raise SystemExit(f"{directory}: need at least 2 captures, found {len(files)}")
    return files[-2], files[-1]


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "input_text"):
                    parts.append(str(item.get("text") or item.get("input_text") or ""))
                elif item.get("type") in ("image_url", "input_image"):
                    parts.append("[IMAGE]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "".join(parts)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _canonical(msg: dict) -> str:
    role = str(msg.get("role") or "")
    body = _content_to_text(msg.get("content"))
    calls = msg.get("tool_calls")
    suffix = ""
    if isinstance(calls, list) and calls:
        suffix = json.dumps(calls, ensure_ascii=False, sort_keys=True)
    return f"<|{role}|>{body}{suffix}"


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _http_json(url: str, payload: dict, timeout: float = 120.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _render_and_tokenize(base_url: str, messages: list[dict]) -> list[int]:
    rendered = _http_json(base_url.rstrip("/") + "/apply-template", {"messages": messages})
    prompt = rendered.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise SystemExit(f"/apply-template returned no prompt: {str(rendered)[:200]}")
    tok = _http_json(
        base_url.rstrip("/") + "/tokenize",
        {"content": prompt, "add_special": False, "parse_special": True},
    )
    tokens = tok.get("tokens")
    if not isinstance(tokens, list):
        raise SystemExit(f"/tokenize returned no tokens: {str(tok)[:200]}")
    return [int(t) for t in tokens]


def _token_lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _snippet(text: str, pos: int, radius: int = 80) -> str:
    lo = max(0, pos - radius)
    hi = min(len(text), pos + radius)
    return text[lo:pos] + " ⟦DIVERGE⟧ " + text[pos:hi]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="*", help="two capture JSON files (A B)")
    ap.add_argument("--dir", help="directory: use the two newest captures")
    ap.add_argument("--base-url", default="http://127.0.0.1:18110", help="llama-server base URL for exact token LCP")
    ap.add_argument("--tokenize", action="store_true", help="compute exact token LCP via /apply-template + /tokenize")
    ap.add_argument("--max-messages", type=int, default=12, help="max differing message summary lines")
    args = ap.parse_args(argv)

    if args.dir:
        if args.captures:
            raise SystemExit("pass either capture files or --dir, not both")
        a_path, b_path = _two_newest(args.dir)
    elif len(args.captures) == 2:
        a_path, b_path = args.captures
    else:
        ap.error("provide two captures or --dir")

    A, B = _load(a_path), _load(b_path)
    ma = A["payload"].get("messages") or []
    mb = B["payload"].get("messages") or []

    print("=" * 78)
    print(f"A: {a_path}")
    print(f"   ts={A.get('ts')} req={A.get('request_id')} model={A.get('upstream_model') or A.get('model')} stream={A.get('stream')}")
    print(f"B: {b_path}")
    print(f"   ts={B.get('ts')} req={B.get('request_id')} model={B.get('upstream_model') or B.get('model')} stream={B.get('stream')}")
    print("=" * 78)

    print(f"\nmessages: A={len(ma)}  B={len(mb)}")
    roles_a = [str(m.get("role") or "") for m in ma]
    roles_b = [str(m.get("role") or "") for m in mb]
    print(f"roles A: {roles_a}")
    print(f"roles B: {roles_b}")

    # message-level: find first index whose canonical form differs
    ca = [_canonical(m) for m in ma]
    cb = [_canonical(m) for m in mb]
    first_diff = None
    for i in range(min(len(ca), len(cb))):
        if ca[i] != cb[i]:
            first_diff = i
            break
    if first_diff is None and len(ca) != len(cb):
        first_diff = min(len(ca), len(cb))

    if first_diff is None:
        print("\nmessage-level: identical message lists")
    else:
        print(f"\nmessage-level: first divergent message index = {first_diff}")
        shown = 0
        for i in range(first_diff, min(max(len(ca), len(cb)), first_diff + args.max_messages)):
            a_msg = ca[i] if i < len(ca) else "<missing>"
            b_msg = cb[i] if i < len(cb) else "<missing>"
            if a_msg == b_msg:
                continue
            tag = "≠" if i == first_diff else " "
            print(f"  {tag} #{i} len A={len(a_msg)} B={len(b_msg)}")
            print(f"      A: {a_msg[:220]!r}")
            print(f"      B: {b_msg[:220]!r}")
            shown += 1
        if shown == 0:
            print("  (difference is only trailing message count)")

    # char-level over canonical concatenation
    text_a = "".join(ca)
    text_b = "".join(cb)
    cpl = _common_prefix(text_a, text_b)
    print(f"\nchar-level: common prefix = {cpl} / min({len(text_a)}, {len(text_b)}) chars")
    if cpl < min(len(text_a), len(text_b)):
        print("  A around divergence: " + _snippet(text_a, cpl))
        print("  B around divergence: " + _snippet(text_b, cpl))

    # replay-through-server token LCP
    if args.tokenize:
        print(f"\ntoken-level (replay via {args.base_url}):")
        try:
            ta = _render_and_tokenize(args.base_url, ma)
            tb = _render_and_tokenize(args.base_url, mb)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to tokenize: {exc}")
            return 2
        lcp = _token_lcp(ta, tb)
        print(f"  tokens A={len(ta)}  B={len(tb)}  lcp={lcp}")
        print(f"  append(B only) = {len(tb) - lcp}   rewind(A only) = {len(ta) - lcp}")
        if lcp < min(len(ta), len(tb)):
            lo = max(0, lcp - 6)
            hi = min(min(len(ta), len(tb)), lcp + 6)
            print(f"  A tokens[{lo}:{hi}] = {ta[lo:hi]}")
            print(f"  B tokens[{lo}:{hi}] = {tb[lo:hi]}")
        print("\n  Compare with the journal:")
        print("    edit/divergence sample (cached/incoming/lcp/reusable/rewind/append/cache_prompt)")
        print("    -> incoming=B, lcp should match the value above, append=B-lcp, rewind=A-lcp")

    return 0


if __name__ == "__main__":
    sys.exit(main())
