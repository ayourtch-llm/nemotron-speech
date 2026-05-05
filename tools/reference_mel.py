#!/usr/bin/env python3
"""
Generate a reference log-mel spectrogram for the test WAV.

Reads the saved Hann window and mel filterbank from the converted safetensors
(so we use the *exact same constants* as the model expects), then runs the
NeMo FilterbankFeatures pipeline by hand:

    audio (f32, 16 kHz) -> dither(off, eval) -> preemphasis(0.97)
        -> STFT(n_fft=512, win=400, hop=160, center=True, reflect-pad)
        -> magnitude^2 -> mel filterbank (saved) -> log(x + 2**-24)

We deliberately do NOT apply normalize_batch (config says `normalize: NA`).

Outputs a numpy file the Rust test will load to compare.
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def load_wav_mono_f32(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path) as w:
        assert w.getnchannels() == 1, "expected mono wav"
        sr = w.getframerate()
        nf = w.getnframes()
        sw = w.getsampwidth()
        raw = w.readframes(nf)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    return x, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--st", required=True, help="path to safetensors with preproc.window/mel_fb")
    ap.add_argument("--out", required=True, help="output .npz with mel + intermediates")
    ap.add_argument("--preemph", type=float, default=0.97)
    ap.add_argument("--log-zero-guard", type=float, default=2.0**-24)
    args = ap.parse_args()

    audio, sr = load_wav_mono_f32(args.wav)
    assert sr == 16000, f"expected 16 kHz; got {sr}"
    print(f"audio: {audio.shape[0]} samples ({audio.shape[0]/sr:.3f} s)")

    # Pull the exact constants the model trained with.
    # safetensors is huge (2.4 GB) — open mmapped so we don't bring it all into memory.
    from safetensors import safe_open

    with safe_open(args.st, framework="pt") as f:
        window = f.get_tensor("preproc.window")
        mel_fb = f.get_tensor("preproc.mel_fb")
    print(f"window: {tuple(window.shape)} ({window.dtype})")
    print(f"mel_fb: {tuple(mel_fb.shape)} ({mel_fb.dtype})")

    win_length = window.shape[0]                     # 400
    n_mels = mel_fb.shape[1]                         # 128
    n_fft_bins = mel_fb.shape[2]                     # 257
    n_fft = (n_fft_bins - 1) * 2                     # 512
    hop_length = 160                                 # 10 ms @ 16 kHz
    print(f"win_length={win_length}, hop_length={hop_length}, n_fft={n_fft}, n_mels={n_mels}")

    x = torch.from_numpy(audio).float()

    # Preemphasis (NeMo default 0.97). x[t] -= preemph * x[t-1]; x[0] kept.
    if args.preemph and args.preemph != 0.0:
        x_pre = torch.cat([x[:1], x[1:] - args.preemph * x[:-1]], dim=0)
    else:
        x_pre = x

    # STFT — center=True with reflect padding is NeMo default (exact_pad=False).
    stft = torch.stft(
        x_pre,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
        normalized=False,
        onesided=True,
    )
    # stft: (n_fft//2+1, T_frames)
    mag2 = stft.real.pow(2) + stft.imag.pow(2)        # (F, T)
    print(f"mag^2: {tuple(mag2.shape)}")

    # Mel: fb is (1, n_mels, F). Multiply -> (n_mels, T).
    mel = (mel_fb.squeeze(0) @ mag2)                  # (n_mels, T)
    print(f"mel pre-log: range [{mel.min().item():.6e}, {mel.max().item():.6e}]")

    # log with additive zero guard (NeMo default).
    log_mel = torch.log(mel + args.log_zero_guard)
    print(f"log_mel: {tuple(log_mel.shape)} range [{log_mel.min().item():.4f}, {log_mel.max().item():.4f}]")

    # Convert STFT real/imag to numpy as well for finer-grained validation.
    np.savez(
        args.out,
        audio=audio,
        x_preemph=x_pre.numpy(),
        stft_real=stft.real.numpy(),
        stft_imag=stft.imag.numpy(),
        mag2=mag2.numpy(),
        mel=mel.numpy(),
        log_mel=log_mel.numpy(),
        # Constants for the Rust side to assert against
        sample_rate=np.int32(sr),
        win_length=np.int32(win_length),
        hop_length=np.int32(hop_length),
        n_fft=np.int32(n_fft),
        n_mels=np.int32(n_mels),
        preemph=np.float32(args.preemph),
        log_zero_guard=np.float32(args.log_zero_guard),
    )
    print(f"wrote {args.out}")

    # Also dump a flat .bin: u32 n_rows, u32 n_cols, then row-major f32 values.
    # Used by the Rust mel test for byte-exact comparison.
    bin_path = Path(args.out).with_suffix(".bin")
    log_mel_np = log_mel.numpy().astype(np.float32, copy=False)
    rows, cols = log_mel_np.shape
    with open(bin_path, "wb") as f:
        f.write(np.uint32(rows).tobytes())
        f.write(np.uint32(cols).tobytes())
        f.write(np.ascontiguousarray(log_mel_np).tobytes())
    print(f"wrote {bin_path} ({rows}x{cols} f32)")


if __name__ == "__main__":
    main()
