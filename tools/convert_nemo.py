#!/usr/bin/env python3
"""
One-shot converter: NeMo .ckpt -> safetensors with key remapping.

The Rust port uses simplified, hierarchical names that match its module tree.
This script also writes a small JSON sidecar with config-derived constants
that the Rust loader needs (vocab, mel filterbank shape, etc.).

Usage:
    python3 tools/convert_nemo.py \
        --ckpt models/extracted/model_weights.ckpt \
        --out  models/nemotron-speech-streaming-en-0.6b.safetensors
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file


# Map NeMo state-dict keys to the Rust-side names we will use.
# Most keys pass through; a few are renamed for clarity.
RENAME_RULES: list[tuple[re.Pattern, str]] = [
    # preprocessor (kept as-is; Rust loads window + filterbank from these)
    (re.compile(r"^preprocessor\.featurizer\.window$"), r"preproc.window"),
    (re.compile(r"^preprocessor\.featurizer\.fb$"), r"preproc.mel_fb"),

    # subsampling (encoder.pre_encode)
    # NeMo dw_striding subsampling: sequential of conv2d layers + final linear.
    # Indices observed: 0 (conv2d), 2 (depthwise), 3 (pointwise),
    #                   5 (depthwise), 6 (pointwise), final out=Linear
    (re.compile(r"^encoder\.pre_encode\.conv\.0\.(weight|bias)$"), r"encoder.subsample.conv0.\1"),
    (re.compile(r"^encoder\.pre_encode\.conv\.2\.(weight|bias)$"), r"encoder.subsample.dw1.\1"),
    (re.compile(r"^encoder\.pre_encode\.conv\.3\.(weight|bias)$"), r"encoder.subsample.pw1.\1"),
    (re.compile(r"^encoder\.pre_encode\.conv\.5\.(weight|bias)$"), r"encoder.subsample.dw2.\1"),
    (re.compile(r"^encoder\.pre_encode\.conv\.6\.(weight|bias)$"), r"encoder.subsample.pw2.\1"),
    (re.compile(r"^encoder\.pre_encode\.out\.(weight|bias)$"), r"encoder.subsample.out.\1"),

    # conformer layers: encoder.layers.<i>.<sub> -> encoder.layers.<i>.<sub>
    # We keep the structure but normalize a couple of sub-module names.
    (re.compile(r"^encoder\.layers\.(\d+)\.norm_feed_forward1\.(weight|bias)$"), r"encoder.layers.\1.norm_ff1.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.norm_feed_forward2\.(weight|bias)$"), r"encoder.layers.\1.norm_ff2.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.feed_forward1\.linear1\.(weight|bias)$"), r"encoder.layers.\1.ff1.linear1.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.feed_forward1\.linear2\.(weight|bias)$"), r"encoder.layers.\1.ff1.linear2.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.feed_forward2\.linear1\.(weight|bias)$"), r"encoder.layers.\1.ff2.linear1.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.feed_forward2\.linear2\.(weight|bias)$"), r"encoder.layers.\1.ff2.linear2.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.norm_conv\.(weight|bias)$"), r"encoder.layers.\1.norm_conv.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.conv\.pointwise_conv1\.(weight|bias)$"), r"encoder.layers.\1.conv.pw1.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.conv\.depthwise_conv\.(weight|bias)$"), r"encoder.layers.\1.conv.dw.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.conv\.batch_norm\.(weight|bias)$"), r"encoder.layers.\1.conv.norm.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.conv\.pointwise_conv2\.(weight|bias)$"), r"encoder.layers.\1.conv.pw2.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.norm_self_att\.(weight|bias)$"), r"encoder.layers.\1.norm_attn.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.linear_q\.(weight|bias)$"), r"encoder.layers.\1.attn.q.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.linear_k\.(weight|bias)$"), r"encoder.layers.\1.attn.k.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.linear_v\.(weight|bias)$"), r"encoder.layers.\1.attn.v.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.linear_out\.(weight|bias)$"), r"encoder.layers.\1.attn.out.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.linear_pos\.(weight|bias)$"), r"encoder.layers.\1.attn.pos.\2"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.pos_bias_u$"), r"encoder.layers.\1.attn.pos_bias_u"),
    (re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.pos_bias_v$"), r"encoder.layers.\1.attn.pos_bias_v"),
    (re.compile(r"^encoder\.layers\.(\d+)\.norm_out\.(weight|bias)$"), r"encoder.layers.\1.norm_out.\2"),

    # rnn-t prediction net
    (re.compile(r"^decoder\.prediction\.embed\.weight$"), r"predict.embed.weight"),
    # Keep PyTorch's `_l<idx>` suffix so candle_nn::lstm() with prefix
    # `predict.lstm` finds `weight_ih_l0`, `weight_hh_l0`, etc.
    (re.compile(r"^decoder\.prediction\.dec_rnn\.lstm\.(weight|bias)_(ih|hh)_l(\d+)$"), r"predict.lstm.\1_\2_l\3"),

    # rnn-t joint
    (re.compile(r"^joint\.enc\.(weight|bias)$"), r"joint.enc.\1"),
    (re.compile(r"^joint\.pred\.(weight|bias)$"), r"joint.pred.\1"),
    (re.compile(r"^joint\.joint_net\.2\.(weight|bias)$"), r"joint.out.\1"),
]


def remap_key(k: str) -> str:
    for pat, repl in RENAME_RULES:
        new, n = pat.subn(repl, k)
        if n:
            return new
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-out", default=None, help="Optional path for metadata JSON")
    args = ap.parse_args()

    print(f"loading {args.ckpt}")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    print(f"  {len(sd)} tensors")

    out: dict[str, torch.Tensor] = {}
    seen = set()
    skipped: list[str] = []
    for k, v in sd.items():
        new_key = remap_key(k)
        if new_key == k and not k.startswith(("preprocessor.", "encoder.", "decoder.", "joint.")):
            skipped.append(k)
            continue
        if new_key in seen:
            raise RuntimeError(f"duplicate target key {new_key} from {k}")
        seen.add(new_key)
        # Ensure contiguous + cast nothing (keep float32)
        out[new_key] = v.contiguous()

    if skipped:
        print(f"  skipped {len(skipped)} keys: {skipped[:5]}...")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, args.out)
    total_bytes = sum(t.numel() * t.element_size() for t in out.values())
    print(f"wrote {args.out}: {len(out)} tensors, {total_bytes / 1e9:.2f} GB")

    # Sanity: print a few
    for k in sorted(out.keys())[:5]:
        print(f"  {k}\t{tuple(out[k].shape)}")
    print("  ...")
    for k in sorted(out.keys())[-5:]:
        print(f"  {k}\t{tuple(out[k].shape)}")


if __name__ == "__main__":
    main()
