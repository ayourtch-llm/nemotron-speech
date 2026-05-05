//! Validate encoder pieces against the PyTorch reference produced by
//! `tools/reference_encoder.py`.
//!
//! Reference binary format (matches `write_bin` in the Python tool):
//!     u32 ndim
//!     u32 shape[0..ndim]
//!     row-major f32 data
//!
//! Usage:
//!     cargo run --bin encoder_check -- \
//!         --mel  tmp/reference_mel.bin \
//!         --st   models/nemotron-speech-streaming-en-0.6b.safetensors \
//!         --stage subsample \
//!         --ref  tmp/ref_subsample.bin

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use nemotron_speech::model::ModelConfig;
use nemotron_speech::model::encoder::{
    ConformerLayer, DwStridingSubsampling, FastConformerEncoder, rel_position_emb,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    mel: PathBuf,
    #[arg(long)]
    st: PathBuf,
    #[arg(long)]
    stage: String,
    #[arg(long = "ref")]
    reference: PathBuf,
    #[arg(long, default_value_t = 1e-3)]
    atol: f32,
    #[arg(long, default_value_t = 1e-4)]
    rtol: f32,
}

fn read_bin(path: &std::path::Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 4 {
        bail!("file too short");
    }
    let ndim = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let mut shape = Vec::with_capacity(ndim);
    let mut off = 4;
    for _ in 0..ndim {
        shape.push(u32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]) as usize);
        off += 4;
    }
    let nelem: usize = shape.iter().product();
    if bytes.len() != off + nelem * 4 {
        bail!(
            "size mismatch: header {:?} expects {} f32, got {}",
            shape,
            nelem,
            (bytes.len() - off) / 4
        );
    }
    let mut data = Vec::with_capacity(nelem);
    let mut i = off;
    while i < bytes.len() {
        data.push(f32::from_le_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
        ]));
        i += 4;
    }
    Ok((shape, data))
}

fn read_mel_bin(path: &std::path::Path) -> Result<(Vec<usize>, Vec<f32>)> {
    // mel.bin uses (rows, cols) — 2 u32 + data, no ndim prefix.
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 8 {
        bail!("mel too short");
    }
    let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let cols = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut data = Vec::with_capacity(rows * cols);
    let mut i = 8;
    while i < bytes.len() {
        data.push(f32::from_le_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
        ]));
        i += 4;
    }
    Ok((vec![rows, cols], data))
}

fn compare(rust: &Tensor, ref_shape: &[usize], ref_data: &[f32], atol: f32) -> Result<()> {
    let dims = rust.dims();
    if dims != ref_shape {
        bail!(
            "shape mismatch: rust {:?} vs ref {:?}",
            dims,
            ref_shape
        );
    }
    let rust_vec: Vec<f32> = rust.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut argmax = 0usize;
    for i in 0..rust_vec.len() {
        let d = (rust_vec[i] - ref_data[i]).abs();
        sum_abs += d as f64;
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
    }
    let mean = sum_abs / rust_vec.len() as f64;
    println!(
        "shape={:?} max_abs={:.3e} (rust={:.4} ref={:.4} at idx {}) mean_abs={:.3e}",
        dims, max_abs, rust_vec[argmax], ref_data[argmax], argmax, mean
    );
    if max_abs > atol {
        bail!("FAIL: max abs {:.3e} exceeds tolerance {:.3e}", max_abs, atol);
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    let device = Device::Cpu;
    let dtype = DType::F32;

    // load mel (n_mels, T)
    let (mel_shape, mel_data) = read_mel_bin(&args.mel)?;
    println!("mel: {:?}", mel_shape);
    let mel = Tensor::from_vec(mel_data, (1, mel_shape[0], mel_shape[1]), &device)?;

    // build var builder over the safetensors
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.st.clone()], dtype, &device)
            .context("loading safetensors")?
    };

    let cfg = ModelConfig::nemotron_06b();
    let (ref_shape, ref_data) = read_bin(&args.reference)?;

    match args.stage.as_str() {
        "subsample" => {
            let sub = DwStridingSubsampling::new(vb.pp("encoder.subsample"), &cfg)
                .map_err(|e| anyhow::anyhow!("{e:#}"))?;
            let out = sub.forward(&mel).map_err(|e| anyhow::anyhow!("{e:#}"))?;
            compare(&out, &ref_shape, &ref_data, args.atol)?;
        }
        "layer0_ff1" | "layer0_attn" | "layer0" | "encoder" => {
            let sub = DwStridingSubsampling::new(vb.pp("encoder.subsample"), &cfg)
                .map_err(|e| anyhow::anyhow!("{e:#}"))?;
            let enc = sub.forward(&mel).map_err(|e| anyhow::anyhow!("{e:#}"))?;
            let (_b, t, _d) = enc.dims3()?;
            let pos = rel_position_emb(t, cfg.d_model, &device, dtype)
                .map_err(|e| anyhow::anyhow!("{e:#}"))?;

            if args.stage == "encoder" {
                let enc_full = FastConformerEncoder::new(vb.pp("encoder"), cfg.clone())
                    .map_err(|e| anyhow::anyhow!("{e:#}"))?;
                let out = enc_full
                    .forward_offline(&mel)
                    .map_err(|e| anyhow::anyhow!("{e:#}"))?;
                compare(&out, &ref_shape, &ref_data, args.atol)?;
            } else {
                let layer = ConformerLayer::new(vb.pp("encoder.layers.0"), &cfg)
                    .map_err(|e| anyhow::anyhow!("{e:#}"))?;
                // For per-stage tests, we currently expose the full ConformerLayer
                // forward — the Python reference must match. Layer-internal
                // bisection is enabled by partial-stage references generated
                // by the reference script.
                if args.stage == "layer0" {
                    let out = layer.forward(&enc, &pos).map_err(|e| anyhow::anyhow!("{e:#}"))?;
                    compare(&out, &ref_shape, &ref_data, args.atol)?;
                } else {
                    bail!(
                        "stage {} requires layer-internal bisection; not yet wired up in Rust binary. \
                         Use stage=layer0 or stage=encoder.",
                        args.stage
                    );
                }
            }
        }
        s => bail!("unknown stage {s}"),
    }
    println!("OK");
    Ok(())
}
