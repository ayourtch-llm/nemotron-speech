//! Streaming transcription: feeds the audio through the encoder chunk by
//! chunk with cache-aware attention, and greedy-decodes after each chunk.
//!
//! For now the audio side runs offline (full mel + full subsample, then we
//! feed the encoded sequence in chunks). The encoder layers themselves use
//! KV + conv caches so the per-chunk math is what real-time streaming
//! would do; only the audio→features→subsample path is still batch.
//! Streaming the front-end is the next step.
//!
//! Usage:
//!     cargo run --release --bin transcribe_streaming -- \
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
use nemotron_speech::model::encoder::{EncoderCache, FastConformerEncoder};
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
    #[arg(long, default_value_t = false)]
    cpu: bool,
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
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            Device::Cpu
        }
    };
    let dtype = DType::F32;
    println!("device: {:?}", device);

    let audio_v = audio::load_audio_mono_16k(&args.audio)?;
    println!(
        "audio: {} samples ({:.2}s)",
        audio_v.len(),
        audio_v.len() as f32 / 16_000.0
    );

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
    let predict =
        PredictNet::new(vb.pp("predict"), &cfg).map_err(|e| anyhow::anyhow!("predict: {e:#}"))?;
    let joint = JointNet::new(vb.pp("joint"), &cfg).map_err(|e| anyhow::anyhow!("joint: {e:#}"))?;
    let tok = Tokenizer::from_file(&args.tok)?;

    // Subsample once (offline).
    let subsampled = encoder
        .subsample
        .forward(&mel_t)
        .map_err(|e| anyhow::anyhow!("subsample: {e:#}"))?;
    let (_b, t_total, _d) = subsampled.dims3()?;
    println!("encoded frames: {}", t_total);

    let chunk_size = cfg.chunk_size_enc_frames();
    let n_chunks = (t_total + chunk_size - 1) / chunk_size;
    println!(
        "chunk_size={} ({} chunks), att_context_size={:?}",
        chunk_size, n_chunks, cfg.att_context_size
    );

    let mut enc_cache = EncoderCache::empty(cfg.n_layers);
    let mut dec = GreedyDecoder::new(
        &predict,
        GreedyDecoderConfig {
            blank_idx: cfg.blank_idx,
            max_symbols_per_step: 10,
        },
        &device,
        dtype,
    )?;

    let mut all_tokens: Vec<u32> = Vec::new();
    print!("text: ");
    use std::io::Write;
    std::io::stdout().flush().ok();

    let t0 = std::time::Instant::now();
    for c in 0..n_chunks {
        let start = c * chunk_size;
        let len = chunk_size.min(t_total - start);
        let chunk = subsampled.narrow(1, start, len)?.contiguous()?;
        let enc_out = encoder
            .forward_layers_chunked(&chunk, &mut enc_cache)
            .map_err(|e| anyhow::anyhow!("chunk {} forward: {e:#}", c))?;
        // Greedy-decode this chunk: state carries across chunks, so simply
        // call decode once per encoded chunk. enc_out is (1, T, d_enc).
        let enc_seq = enc_out.squeeze(0)?;
        let prev_len = all_tokens.len();
        dec.decode(&enc_seq, &predict, &joint, &mut all_tokens)?;
        if all_tokens.len() > prev_len {
            // Emit the new tokens incrementally — detokenize the prefix
            // and print whatever's new since the last emission.
            let cur_text = tok.detokenize(&all_tokens)?;
            // Compute previous text and diff.
            let prev_text = if prev_len == 0 {
                String::new()
            } else {
                tok.detokenize(&all_tokens[..prev_len])?
            };
            let new_text = cur_text.strip_prefix(&prev_text).unwrap_or(&cur_text);
            print!("{}", new_text);
            std::io::stdout().flush().ok();
        }
    }
    println!();
    println!("\ntotal: {} tokens in {:?}", all_tokens.len(), t0.elapsed());
    Ok(())
}
