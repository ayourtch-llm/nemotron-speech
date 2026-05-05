//! Live transcription driven by an `AudioSource`. Works the same way for
//! file input, microphone (with `--features mic`), and (eventually) UDP
//! packets — they all implement the same trait.
//!
//! Usage:
//!     # file (offline, but driven through the streaming pipeline)
//!     cargo run --release --bin transcribe_live -- \
//!         --st models/nemotron-speech-streaming-en-0.6b.safetensors \
//!         --tok models/tokenizer.model \
//!         --audio tmp/small-test.wav
//!
//!     # microphone
//!     cargo run --release --features mic --bin transcribe_live -- \
//!         --st  models/nemotron-speech-streaming-en-0.6b.safetensors \
//!         --tok models/tokenizer.model \
//!         --mic
//!     # speak; ctrl-C to stop.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use nemotron_speech::audio::load_audio_mono_16k;
use nemotron_speech::audio_source::{AudioSource, FileChunkSource};
#[cfg(feature = "mic")]
use nemotron_speech::audio_source::mic::MicSource;
use nemotron_speech::features::{MelConfig, MelExtractor};
use nemotron_speech::model::ModelConfig;
use nemotron_speech::model::encoder::FastConformerEncoder;
use nemotron_speech::model::joint::JointNet;
use nemotron_speech::model::predict::PredictNet;
use nemotron_speech::streaming::StreamingPipeline;
use nemotron_speech::tokenizer::Tokenizer;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    st: PathBuf,
    #[arg(long)]
    tok: PathBuf,
    #[arg(long, conflicts_with = "mic")]
    audio: Option<PathBuf>,
    /// Read from the default microphone (requires `--features mic`).
    #[arg(long, default_value_t = false)]
    mic: bool,
    #[arg(long, default_value_t = false)]
    cpu: bool,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
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
        #[cfg(not(feature = "metal"))]
        {
            Device::Cpu
        }
    };
    let dtype = DType::F32;
    eprintln!("device: {:?}", device);

    let mel_cfg = MelConfig::nemotron_default();
    let mel = MelExtractor::from_safetensors(&args.st, mel_cfg.clone())?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.st.clone()], dtype, &device)
            .context("loading safetensors")?
    };
    let cfg = ModelConfig::nemotron_06b();
    let encoder = FastConformerEncoder::new(vb.pp("encoder"), cfg.clone())
        .map_err(|e| anyhow::anyhow!("encoder: {e:#}"))?;
    let predict = PredictNet::new(vb.pp("predict"), &cfg)
        .map_err(|e| anyhow::anyhow!("predict: {e:#}"))?;
    let joint = JointNet::new(vb.pp("joint"), &cfg)
        .map_err(|e| anyhow::anyhow!("joint: {e:#}"))?;
    let tok = Tokenizer::from_file(&args.tok)?;

    let mut pipe = StreamingPipeline::new(
        encoder, predict, joint, mel, mel_cfg, cfg, device, dtype,
    )?;

    let mut source: Box<dyn AudioSource> = match (&args.audio, args.mic) {
        (Some(p), false) => {
            let samples = load_audio_mono_16k(p)?;
            // Feed in 320-sample (~20 ms) chunks to exercise the streaming
            // advance logic; the pipeline batches up internally.
            Box::new(FileChunkSource::new(samples, 320))
        }
        #[cfg(feature = "mic")]
        (None, true) => Box::new(MicSource::open_default(320)?),
        #[cfg(not(feature = "mic"))]
        (None, true) => {
            anyhow::bail!("rebuild with --features mic to use microphone input");
        }
        (None, false) => anyhow::bail!("specify --audio <file> or --mic"),
        (Some(_), true) => unreachable!(),
    };

    eprintln!("listening... (Ctrl-C to stop)");
    std::io::stderr().flush().ok();

    loop {
        match source.next_chunk().await? {
            None => break,
            Some(chunk) => {
                pipe.push_audio(&chunk.samples);
                if chunk.is_final {
                    pipe.finish();
                }
                let mut chunk_text = String::new();
                while let Some(new_tokens) = pipe.try_advance()? {
                    if !new_tokens.is_empty() {
                        let total = &pipe.all_tokens;
                        let prev_len = total.len() - new_tokens.len();
                        let prev_text = if prev_len == 0 {
                            String::new()
                        } else {
                            tok.detokenize(&total[..prev_len])?
                        };
                        let cur_text = tok.detokenize(total)?;
                        let new_text = cur_text.strip_prefix(&prev_text).unwrap_or(&cur_text);
                        chunk_text.push_str(new_text);
                    }
                }
                if !chunk_text.is_empty() {
                    eprintln!("[chunk] {}", chunk_text);
                    std::io::stderr().flush().ok();
                }
            }
        }
    }
    eprintln!();
    Ok(())
}
