//! Probe: dump per-LAYER mean-pooled encoder embeddings for a WAV, for all 24
//! FastConformer ConformerLayers. embed_dump only sees the FINAL (speaker-
//! invariant) layer; this exposes every layer so we can find where speaker
//! identity lives. Output file layout: [n_layers:u32_le][dim:u32_le] then
//! n_layers*dim f32_le (each layer mean-pooled over time, L2-normalized).
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use nemotron_speech::audio;
use nemotron_speech::features::{MelConfig, MelExtractor};
use nemotron_speech::model::ModelConfig;
use nemotron_speech::model::encoder::FastConformerEncoder;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    audio: PathBuf,
    #[arg(long)]
    st: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = false)]
    cpu: bool,
    /// statistics pooling: concat [mean; std] per layer (2*d_model) instead of mean only.
    #[arg(long, default_value_t = false)]
    stats: bool,
    /// if >=0, dump the FULL per-frame activations (T x d_model) for this single
    /// layer instead of pooled per-layer embeddings. Output: [T:u32][D:u32] T*D f32.
    #[arg(long, default_value_t = -1)]
    frames_layer: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(feature = "metal"))]
        {
            Device::Cpu
        }
    };
    let dtype = DType::F32;

    let audio_v = audio::load_audio_mono_16k(&args.audio)?;
    let mel_cfg = MelConfig::nemotron_default();
    let mut mel = MelExtractor::from_safetensors(&args.st, mel_cfg.clone())?;
    let log_mel = mel.forward(&audio_v);
    let n_frames = log_mel.len() / mel_cfg.n_mels;
    let mel_t = Tensor::from_vec(log_mel, (1, mel_cfg.n_mels, n_frames), &device)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.st.clone()], dtype, &device)
            .context("loading safetensors")?
    };
    let cfg = ModelConfig::nemotron_06b();
    let encoder = FastConformerEncoder::new(vb.pp("encoder"), cfg.clone())
        .map_err(|e| anyhow::anyhow!("encoder: {e:#}"))?;
    let layers = encoder
        .forward_layers(&mel_t, false)
        .map_err(|e| anyhow::anyhow!("encoder forward_layers: {e:#}"))?;

    // Per-frame dump mode for a single layer (for windowed / per-word tagging).
    if args.frames_layer >= 0 {
        let idx = args.frames_layer as usize;
        let enc = &layers[idx]; // (1, T, D)
        let frames = enc.i(0)?; // (T, D)
        let (t, d) = frames.dims2()?;
        let data = frames.to_vec2::<f32>()?;
        let mut f = std::fs::File::create(&args.out)?;
        f.write_all(&(t as u32).to_le_bytes())?;
        f.write_all(&(d as u32).to_le_bytes())?;
        for row in &data {
            for x in row {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        eprintln!("frames layer {} -> {} frames x {} dims -> {}", idx, t, d, args.out.display());
        return Ok(());
    }

    let n_layers = layers.len() as u32;
    let out_dim = if args.stats { (cfg.d_model * 2) as u32 } else { cfg.d_model as u32 };
    let mut f = std::fs::File::create(&args.out)?;
    f.write_all(&n_layers.to_le_bytes())?;
    f.write_all(&out_dim.to_le_bytes())?;
    for (li, enc) in layers.iter().enumerate() {
        // mean-pool over time -> (1, D); optionally append std (statistics pooling).
        let mean = enc.mean(1)?;
        let mut v = mean.flatten_all()?.to_vec1::<f32>()?;
        if args.stats {
            let mean2 = enc.sqr()?.mean(1)?;
            let var = (mean2 - mean.sqr()?)?;
            let std = (var + 1e-8)?.sqrt()?;
            v.extend(std.flatten_all()?.to_vec1::<f32>()?);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &v {
            f.write_all(&(x / norm).to_le_bytes())?;
        }
        if li == 0 {
            eprintln!("n_layers={} out_dim={} frames={} stats={}", n_layers, out_dim, n_frames, args.stats);
        }
    }
    eprintln!("wrote {} layers x {} dims -> {}", n_layers, out_dim, args.out.display());
    Ok(())
}
