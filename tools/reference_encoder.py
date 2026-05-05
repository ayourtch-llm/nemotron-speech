#!/usr/bin/env python3
"""
PyTorch reference for selected encoder pieces. Lets the Rust port validate
each module against an exact reference computed from the same weights.

Currently exports:
  --stage subsample     dwstriding x8 + final linear
  --stage layer0_ff1    Macaron FFN-1 of layer 0
  --stage layer0_attn   Self-attention block of layer 0 (input is post-FF1)
  --stage layer0        Full layer 0 (FF1 -> Attn -> Conv -> FF2 -> LN_out)
  --stage encoder       All 24 layers
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


# ----- helpers --------------------------------------------------------------

def st_load(path: str) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def write_bin(path: Path, t: torch.Tensor):
    """Flat dump: u32 ndim, u32 shape[0..ndim-1], then row-major f32 data."""
    arr = t.detach().cpu().float().contiguous().numpy()
    with open(path, "wb") as f:
        f.write(np.uint32(arr.ndim).tobytes())
        for d in arr.shape:
            f.write(np.uint32(d).tobytes())
        f.write(arr.tobytes())
    print(f"wrote {path}: shape={tuple(arr.shape)}")


# ----- modules --------------------------------------------------------------

class CausalConv2d(nn.Module):
    """k=3, s=2 with asymmetric pad (k-1, s-1) = (2, 1) on H and W."""

    def __init__(self, w: torch.Tensor, b: torch.Tensor, groups: int = 1):
        super().__init__()
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, pad=(2, 1, 2, 1))
        return F.conv2d(x, self.weight, self.bias, stride=2, groups=self.groups)


class Pointwise2d(nn.Module):
    def __init__(self, w: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias)


class Subsample(nn.Module):
    def __init__(self, sd: dict[str, torch.Tensor]):
        super().__init__()
        self.conv0 = CausalConv2d(sd["encoder.subsample.conv0.weight"], sd["encoder.subsample.conv0.bias"])
        self.dw1 = CausalConv2d(sd["encoder.subsample.dw1.weight"], sd["encoder.subsample.dw1.bias"], groups=256)
        self.pw1 = Pointwise2d(sd["encoder.subsample.pw1.weight"].unsqueeze(-1) if sd["encoder.subsample.pw1.weight"].dim() == 3 else sd["encoder.subsample.pw1.weight"], sd["encoder.subsample.pw1.bias"])
        self.dw2 = CausalConv2d(sd["encoder.subsample.dw2.weight"], sd["encoder.subsample.dw2.bias"], groups=256)
        self.pw2 = Pointwise2d(sd["encoder.subsample.pw2.weight"].unsqueeze(-1) if sd["encoder.subsample.pw2.weight"].dim() == 3 else sd["encoder.subsample.pw2.weight"], sd["encoder.subsample.pw2.bias"])
        self.out_w = nn.Parameter(sd["encoder.subsample.out.weight"])
        self.out_b = nn.Parameter(sd["encoder.subsample.out.bias"])

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, n_mels, T)
        x = mel.transpose(1, 2).unsqueeze(1)  # (B, 1, T, n_mels)
        x = F.relu(self.conv0(x))
        x = self.dw1(x)
        x = F.relu(self.pw1(x))
        x = self.dw2(x)
        x = F.relu(self.pw2(x))
        # (B, C, T', F') -> (B, T', C*F')
        b, c, tp, fp = x.size()
        x = x.transpose(1, 2).reshape(b, tp, c * fp)
        x = F.linear(x, self.out_w, self.out_b)
        return x


def make_pos_emb(seq_len: int, d_model: int, device, dtype) -> torch.Tensor:
    positions = torch.arange(seq_len - 1, -seq_len, -1, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / d_model)
    )
    pe = torch.zeros(positions.size(0), d_model, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.unsqueeze(0).to(dtype)


class FeedForward(nn.Module):
    def __init__(self, sd: dict[str, torch.Tensor], prefix: str):
        super().__init__()
        self.l1_w = nn.Parameter(sd[f"{prefix}.linear1.weight"])
        self.l2_w = nn.Parameter(sd[f"{prefix}.linear2.weight"])

    def forward(self, x):
        x = F.linear(x, self.l1_w)  # no bias
        x = F.silu(x)               # Swish
        x = F.linear(x, self.l2_w)
        return x


class RelPosMha(nn.Module):
    def __init__(self, sd, prefix, n_heads=8, d_head=128):
        super().__init__()
        self.h = n_heads
        self.dk = d_head
        self.q = nn.Parameter(sd[f"{prefix}.q.weight"])
        self.k = nn.Parameter(sd[f"{prefix}.k.weight"])
        self.v = nn.Parameter(sd[f"{prefix}.v.weight"])
        self.out = nn.Parameter(sd[f"{prefix}.out.weight"])
        self.pos = nn.Parameter(sd[f"{prefix}.pos.weight"])
        self.bu = nn.Parameter(sd[f"{prefix}.pos_bias_u"])  # (h, dk)
        self.bv = nn.Parameter(sd[f"{prefix}.pos_bias_v"])

    def forward(self, x, pos_emb):
        B, T, D = x.shape
        h, dk = self.h, self.dk

        def proj(t, w):
            return F.linear(t, w).reshape(B, T, h, dk).transpose(1, 2).contiguous()

        q = proj(x, self.q)
        k = proj(x, self.k)
        v = proj(x, self.v)

        # pos: (1, 2T-1, D) -> (1, h, 2T-1, dk)
        p = F.linear(pos_emb, self.pos).reshape(1, pos_emb.size(1), h, dk).transpose(1, 2).contiguous()

        q_u = q + self.bu.view(1, h, 1, dk)
        q_v = q + self.bv.view(1, h, 1, dk)

        ac = torch.matmul(q_u, k.transpose(-2, -1))                        # (B, h, T, T)
        bd = torch.matmul(q_v, p.transpose(-2, -1))                        # (B, h, T, 2T-1)
        # rel_shift trick (NeMo style)
        bd = F.pad(bd, (1, 0))                                              # (B, h, T, 2T)
        bd = bd.view(B, h, -1, T)                                           # (B, h, 2T, T)
        bd = bd[:, :, 1:].view(B, h, T, 2 * T - 1)                          # (B, h, T, 2T-1)
        bd = bd[:, :, :, :T]                                                # (B, h, T, T)

        scores = (ac + bd) / math.sqrt(dk)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)                                         # (B, h, T, dk)
        ctx = ctx.transpose(1, 2).contiguous().reshape(B, T, h * dk)
        return F.linear(ctx, self.out)


class ConvModule(nn.Module):
    def __init__(self, sd, prefix, d_model=1024, k=9):
        super().__init__()
        self.d = d_model
        self.k = k
        # use_bias=false for the encoder, so pw1/dw/pw2 have no bias.
        self.pw1_w = nn.Parameter(sd[f"{prefix}.pw1.weight"])  # (2d, d, 1)
        self.dw_w = nn.Parameter(sd[f"{prefix}.dw.weight"])    # (d, 1, k)
        self.pw2_w = nn.Parameter(sd[f"{prefix}.pw2.weight"])
        self.norm_w = nn.Parameter(sd[f"{prefix}.norm.weight"])
        self.norm_b = nn.Parameter(sd[f"{prefix}.norm.bias"])

    def forward(self, x):
        # x: (B, T, d) -> (B, d, T)
        x = x.transpose(1, 2)
        x = F.conv1d(x, self.pw1_w)                                 # (B, 2d, T)
        x = F.glu(x, dim=1)                                          # (B, d, T)
        x = F.pad(x, (self.k - 1, 0))                                # causal pad on time
        x = F.conv1d(x, self.dw_w, groups=self.d)                    # (B, d, T)
        # LayerNorm on channel dim
        x = x.transpose(1, 2)
        x = F.layer_norm(x, (self.d,), self.norm_w, self.norm_b)
        x = F.silu(x)
        x = x.transpose(1, 2)
        x = F.conv1d(x, self.pw2_w)                                  # (B, d, T)
        return x.transpose(1, 2)


class ConformerLayer(nn.Module):
    def __init__(self, sd, idx, d=1024, d_ff=4096):
        super().__init__()
        p = f"encoder.layers.{idx}"
        self.norm_ff1_w = nn.Parameter(sd[f"{p}.norm_ff1.weight"])
        self.norm_ff1_b = nn.Parameter(sd[f"{p}.norm_ff1.bias"])
        self.ff1 = FeedForward(sd, f"{p}.ff1")
        self.norm_attn_w = nn.Parameter(sd[f"{p}.norm_attn.weight"])
        self.norm_attn_b = nn.Parameter(sd[f"{p}.norm_attn.bias"])
        self.attn = RelPosMha(sd, f"{p}.attn")
        self.norm_conv_w = nn.Parameter(sd[f"{p}.norm_conv.weight"])
        self.norm_conv_b = nn.Parameter(sd[f"{p}.norm_conv.bias"])
        self.conv = ConvModule(sd, f"{p}.conv")
        self.norm_ff2_w = nn.Parameter(sd[f"{p}.norm_ff2.weight"])
        self.norm_ff2_b = nn.Parameter(sd[f"{p}.norm_ff2.bias"])
        self.ff2 = FeedForward(sd, f"{p}.ff2")
        self.norm_out_w = nn.Parameter(sd[f"{p}.norm_out.weight"])
        self.norm_out_b = nn.Parameter(sd[f"{p}.norm_out.bias"])
        self.d = d

    def forward(self, x, pos_emb):
        r = x
        y = F.layer_norm(x, (self.d,), self.norm_ff1_w, self.norm_ff1_b)
        x = r + 0.5 * self.ff1(y)

        r = x
        y = F.layer_norm(x, (self.d,), self.norm_attn_w, self.norm_attn_b)
        x = r + self.attn(y, pos_emb)

        r = x
        y = F.layer_norm(x, (self.d,), self.norm_conv_w, self.norm_conv_b)
        x = r + self.conv(y)

        r = x
        y = F.layer_norm(x, (self.d,), self.norm_ff2_w, self.norm_ff2_b)
        x = r + 0.5 * self.ff2(y)

        return F.layer_norm(x, (self.d,), self.norm_out_w, self.norm_out_b)


# ----- entry ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel-bin", required=True, help="reference_mel.bin (rows=128, cols=T)")
    ap.add_argument("--st", required=True, help="converted safetensors")
    ap.add_argument("--stage", required=True, choices=["subsample", "layer0_ff1", "layer0_attn", "layer0", "encoder"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # load mel
    with open(args.mel_bin, "rb") as f:
        rows = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        cols = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(rows, cols)
    mel = torch.from_numpy(data).unsqueeze(0).contiguous()  # (1, n_mels, T)
    print(f"mel: {tuple(mel.shape)}")

    sd = st_load(args.st)
    # Strip the singleton from preproc.mel_fb if present (irrelevant here).
    torch.set_grad_enabled(False)

    sub = Subsample(sd).eval()
    enc = sub(mel)  # (1, T_out, 1024)
    print(f"after subsample: {tuple(enc.shape)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stage == "subsample":
        write_bin(out_path, enc)
        return

    # build pos emb for encoder length
    T = enc.size(1)
    pos = make_pos_emb(T, 1024, enc.device, enc.dtype)
    print(f"pos_emb: {tuple(pos.shape)}")

    if args.stage in ("layer0_ff1", "layer0_attn", "layer0", "encoder"):
        layer0 = ConformerLayer(sd, 0).eval()

        if args.stage == "layer0_ff1":
            r = enc
            y = F.layer_norm(enc, (1024,), layer0.norm_ff1_w, layer0.norm_ff1_b)
            x = r + 0.5 * layer0.ff1(y)
            write_bin(out_path, x)
            return

        if args.stage == "layer0_attn":
            # Output of FF1 -> Attn block (post-residual, before Conv block).
            r = enc
            y = F.layer_norm(enc, (1024,), layer0.norm_ff1_w, layer0.norm_ff1_b)
            x = r + 0.5 * layer0.ff1(y)

            r = x
            y = F.layer_norm(x, (1024,), layer0.norm_attn_w, layer0.norm_attn_b)
            x = r + layer0.attn(y, pos)
            write_bin(out_path, x)
            return

        if args.stage == "layer0":
            x = layer0(enc, pos)
            write_bin(out_path, x)
            return

        # full encoder
        x = enc
        layers = [ConformerLayer(sd, i).eval() for i in range(24)]
        for i, lyr in enumerate(layers):
            x = lyr(x, pos)
        write_bin(out_path, x)
        return


if __name__ == "__main__":
    main()
