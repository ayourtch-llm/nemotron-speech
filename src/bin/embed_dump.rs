//! Probe: dump a mean-pooled, L2-normalized FastConformer ENCODER embedding for
//! a WAV. Used to test whether the ASR encoder's representation separates
//! speakers (i.e. whether it can serve as an accidental voice fingerprint) —
//! ASR encoders are trained to be speaker-INVARIANT, so this is expected to be
//! weak; this binary lets us measure how weak.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
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
    let enc = encoder
        .forward_full(&mel_t, false)
        .map_err(|e| anyhow::anyhow!("encoder forward: {e:#}"))?; // (1, T, D)

    // Mean-pool over time -> (1, D), then L2-normalize.
    let pooled = enc.mean(1)?;
    let v = pooled.flatten_all()?.to_vec1::<f32>()?;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    let vn: Vec<f32> = v.iter().map(|x| x / norm).collect();

    let mut f = std::fs::File::create(&args.out)?;
    for x in &vn {
        f.write_all(&x.to_le_bytes())?;
    }
    eprintln!("wrote {} dims -> {}", vn.len(), args.out.display());
    Ok(())
}
