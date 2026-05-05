//! End-to-end offline transcription:
//!     audio file -> mel -> encoder -> RNN-T greedy -> text.
//!
//! Streaming + cache support is intentionally out of scope here; this binary
//! exists to prove the offline path produces sensible English on a short
//! clip, which is the gating step before adding chunked masking.
//!
//! Usage:
//!     cargo run --release --bin transcribe -- \
//!         --audio tmp/small-test.m4a \
//!         --st    models/nemotron-speech-streaming-en-0.6b.safetensors \
//!         --tok   models/tokenizer.model

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use nemotron_speech::audio;
use nemotron_speech::features::{MelConfig, MelExtractor};
use nemotron_speech::model::ModelConfig;
use nemotron_speech::model::encoder::FastConformerEncoder;
use nemotron_speech::model::greedy::{GreedyDecoder, GreedyDecoderConfig};
use nemotron_speech::model::joint::JointNet;
use nemotron_speech::model::predict::PredictNet;
use nemotron_speech::tokenizer::Tokenizer;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    audio: PathBuf,
    #[arg(long)]
    st: PathBuf,
    #[arg(long)]
    tok: PathBuf,
    /// Force CPU even if Metal/CUDA features are enabled.
    #[arg(long, default_value_t = false)]
    cpu: bool,
    /// Apply the chunked-limited attention mask the model was trained with.
    /// For utterances shorter than (left_chunks+1) * chunk_size encoder
    /// frames, this is a no-op and output should be byte-identical to the
    /// full-attention path.
    #[arg(long, default_value_t = false)]
    chunked_mask: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    let device = if args.cpu {
        Device::Cpu
    } else {
        // Try Metal, fall back to CPU. CUDA path engages with the `cuda` feature.
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
    println!("device: {:?}, dtype: {:?}", device, dtype);

    // Audio
    let t0 = std::time::Instant::now();
    let audio_v = audio::load_audio_mono_16k(&args.audio)?;
    println!(
        "audio: {} samples ({:.2}s) loaded in {:?}",
        audio_v.len(),
        audio_v.len() as f32 / 16_000.0,
        t0.elapsed()
    );

    // Mel
    let mel_cfg = MelConfig::nemotron_default();
    let mut mel = MelExtractor::from_safetensors(&args.st, mel_cfg.clone())?;
    let t0 = std::time::Instant::now();
    let log_mel = mel.forward(&audio_v);
    println!("mel: computed in {:?}", t0.elapsed());

    let n_frames = log_mel.len() / mel_cfg.n_mels;
    let mel_t = Tensor::from_vec(log_mel, (1, mel_cfg.n_mels, n_frames), &device)?;

    // Build VB for the model
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.st.clone()], dtype, &device)
            .context("loading safetensors")?
    };

    let cfg = ModelConfig::nemotron_06b();

    println!("loading encoder...");
    let t0 = std::time::Instant::now();
    let encoder = FastConformerEncoder::new(vb.pp("encoder"), cfg.clone())
        .map_err(|e| anyhow::anyhow!("encoder: {e:#}"))?;
    println!("encoder loaded in {:?}", t0.elapsed());

    println!("loading predict + joint...");
    let predict = PredictNet::new(vb.pp("predict"), &cfg)
        .map_err(|e| anyhow::anyhow!("predict: {e:#}"))?;
    let joint = JointNet::new(vb.pp("joint"), &cfg)
        .map_err(|e| anyhow::anyhow!("joint: {e:#}"))?;

    // Encoder forward
    println!("encoder forward...");
    let t0 = std::time::Instant::now();
    let enc_out = encoder
        .forward_full(&mel_t, args.chunked_mask)
        .map_err(|e| anyhow::anyhow!("encoder forward: {e:#}"))?;
    println!("encoder forward took {:?}; output shape {:?}", t0.elapsed(), enc_out.dims());

    // Greedy decode
    let mut dec = GreedyDecoder::new(
        &predict,
        GreedyDecoderConfig {
            blank_idx: cfg.blank_idx,
            max_symbols_per_step: 10,
        },
        &device,
        dtype,
    )?;

    // enc_out is (1, T, d_model). Squeeze batch.
    let enc_seq = enc_out.squeeze(0)?;
    let mut tokens: Vec<u32> = Vec::new();
    let t0 = std::time::Instant::now();
    dec.decode(&enc_seq, &predict, &joint, &mut tokens)?;
    println!(
        "greedy decode produced {} tokens in {:?}",
        tokens.len(),
        t0.elapsed()
    );
    println!("tokens: {:?}", tokens);

    // Detokenize
    let tok = Tokenizer::from_file(&args.tok)?;
    let text = tok.detokenize(&tokens)?;
    println!("\n=== transcription ===\n{}\n", text);
    Ok(())
}
