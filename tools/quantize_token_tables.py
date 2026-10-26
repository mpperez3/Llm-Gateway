#!/usr/bin/env python3
"""Requantize the token tables (``lm_head`` + ``embed_tokens``) of a
compressed-tensors W4A16 checkpoint to int8, in place.

Why: published W4A16 quants leave both matrices in bf16. For Qwen3.8-27B the
vocabulary is 248320 x 5120, so each one is 2.37 GiB — and ``lm_head`` is read
on *every* decode step. int8 halves both (and, for the embedding, halves the
``gather`` traffic). On a 24 GB card that ~2.4 GiB is the difference between
"262144 context + CUDA graphs" and "no".

Format produced (what vLLM's compressed-tensors loader expects):
    <key>_packed  int32 (4 int8 per word, packed along dim 1)
    <key>_scale   fp16 (linear) / bf16 (embedding), one per group
    <key>_shape   int64 [out, in]
plus two extra ``config_groups`` entries whose regex targets select exactly
these tensors, and the ``mtp.*`` modules added to ``ignore`` (they are stored
bf16 and would otherwise be picked up by the Linear group).

Requires the vLLM patches in ``llamacpp_stack/bundle/patches-vllm`` for the
embedding path to actually be used.

Derived from the MIT-licensed scripts in
https://github.com/xvus/syv-ai-qwen38-27b-rtx3090 (prepare/quant_lm_head.py,
prepare/quant_embed.py); merged so both keys can live in one shard.

Usage:
    python tools/quantize_token_tables.py /var/llamacpp_models/<ckpt> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

GROUP = 128
BITS = 8
QMAX = 127
MAX_REL_ERROR = 0.01

MTP_IGNORE = (
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
)


def quantize_matrix(torch, weight, group: int, bits: int):
    qmax = 2 ** (bits - 1) - 1
    out_f, in_f = weight.shape
    if in_f % group:
        raise SystemExit(f"in_features {in_f} is not a multiple of {group}")
    grouped = weight.to(torch.float32).reshape(out_f, in_f // group, group)
    scale = torch.clamp(grouped.abs().amax(dim=-1, keepdim=True) / qmax, min=1e-10)
    q = torch.clamp(torch.round(grouped / scale), -qmax - 1, qmax).to(torch.int8)
    q = q.reshape(out_f, in_f)
    deq = (q.reshape(out_f, -1, group).to(torch.float32) * scale).reshape(out_f, in_f)
    err = ((deq - weight.to(torch.float32)).norm() / weight.to(torch.float32).norm()).item()
    return q, scale.squeeze(-1), err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--bits", type=int, default=BITS)
    ap.add_argument("--embed-bits", type=int, default=None, help="override --bits for embed_tokens")
    ap.add_argument("--lm-head-bits", type=int, default=None, help="override --bits for lm_head")
    ap.add_argument("--group", type=int, default=GROUP)
    ap.add_argument("--dry-run", action="store_true", help="measure error, write nothing")
    args = ap.parse_args()

    import torch
    from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
    from safetensors import safe_open
    from safetensors.torch import save_file

    d = args.model_dir
    index_path = d / "model.safetensors.index.json"
    config_path = d / "config.json"
    idx = json.loads(index_path.read_text())
    wm = idx["weight_map"]
    cfg = json.loads(config_path.read_text())
    qc = cfg["quantization_config"]

    embed_keys = [k for k in wm if k.endswith("embed_tokens.weight")]
    targets: list[tuple[str, str]] = []  # (key, "linear"|"embedding")
    if "lm_head.weight" in wm:
        targets.append(("lm_head.weight", "linear"))
    for k in embed_keys:
        targets.append((k, "embedding"))
    if not targets:
        raise SystemExit("no lm_head.weight / *embed_tokens.weight in the index")

    pending = {k: kind for k, kind in targets if f"{k[:-len('.weight')]}.weight_packed" not in wm}
    if not pending:
        print("nothing to do: token tables already quantized")
        return 0

    def bits_for(kind: str) -> int:
        if kind == "embedding" and args.embed_bits:
            return args.embed_bits
        if kind == "linear" and args.lm_head_bits:
            return args.lm_head_bits
        return args.bits

    for key, kind in pending.items():
        shard = wm[key]
        path = d / shard
        with safe_open(path, framework="pt") as f:
            meta = f.metadata()
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        weight = tensors[key]
        bits = bits_for(kind)
        q, scale, err = quantize_matrix(torch, weight, args.group, bits)
        print(f"{key}: {tuple(weight.shape)} rel_err={err:.4f} ({kind})")
        if err >= MAX_REL_ERROR:
            raise SystemExit(f"quantization error too high for {key} ({err:.4f})")
        if args.dry_run:
            continue

        stem = key[: -len(".weight")]
        tensors.pop(key)
        tensors[f"{stem}.weight_packed"] = pack_to_int32(q, bits, packed_dim=1).contiguous()
        scale_dtype = torch.bfloat16 if kind == "embedding" else torch.float16
        tensors[f"{stem}.weight_scale"] = scale.to(scale_dtype).contiguous()
        tensors[f"{stem}.weight_shape"] = torch.tensor(list(weight.shape), dtype=torch.int64)

        if not (path.with_suffix(path.suffix + ".bak-tokentables")).exists():
            shutil.copy(path, path.with_suffix(path.suffix + ".bak-tokentables"))
        save_file(tensors, path, metadata=meta or {"format": "pt"})
        print(f"  wrote {shard}")

        del wm[key]
        for suffix in ("weight_packed", "weight_scale", "weight_shape"):
            wm[f"{stem}.{suffix}"] = shard

    if args.dry_run:
        print("dry-run: nothing written")
        return 0

    if not index_path.with_suffix(index_path.suffix + ".bak-tokentables").exists():
        shutil.copy(index_path, index_path.with_suffix(index_path.suffix + ".bak-tokentables"))
    index_path.write_text(json.dumps(idx, indent=2))

    # Config: stop ignoring the tables, ignore the bf16 MTP module, and add one
    # group per table so packed int8 tensors are actually selected.
    ignore = [i for i in qc.get("ignore", [])]
    for name in ("lm_head",):
        ignore = [i for i in ignore if i != name]
    group0 = qc["config_groups"]["group_0"]
    existing_targets = {tuple(g.get("targets", [])) for g in qc["config_groups"].values()}
    for key, kind in pending.items():
        regex = f"re:.*{key.split('.')[-2]}$" if kind == "embedding" else "re:.*lm_head$"
        if (regex,) in existing_targets:
            continue
        group = json.loads(json.dumps(group0))
        group["targets"] = [regex]
        group["weights"]["num_bits"] = bits_for(kind)
        group["weights"]["group_size"] = args.group
        group["weights"]["strategy"] = "group"
        _used = [int(m.group(1)) for k in qc["config_groups"] if (m := __import__("re").fullmatch(r"group_(\d+)", k))]
        name = f"group_{(max(_used) + 1) if _used else 0}"
        qc["config_groups"][name] = group
    for module in MTP_IGNORE:
        if module not in ignore:
            ignore.append(module)
    qc["ignore"] = ignore
    if not config_path.with_suffix(config_path.suffix + ".bak-tokentables").exists():
        shutil.copy(config_path, config_path.with_suffix(config_path.suffix + ".bak-tokentables"))
    config_path.write_text(json.dumps(cfg, indent=2))
    print("config.json updated (groups added, mtp.* ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
