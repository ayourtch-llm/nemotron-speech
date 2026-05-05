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
use nemotron_speech::audio_source::{AudioSource, FileChunkSource, UdpSource};
#[cfg(feature = "mic")]
use nemotron_speech::audio_source::mic::MicSource;
use nemotron_speech::features::{IncrementalMelExtractor, MelConfig};
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
    #[arg(long, conflicts_with_all = ["mic", "udp_listen"])]
    audio: Option<PathBuf>,
    /// Read from the default microphone (requires `--features mic`).
    #[arg(long, default_value_t = false, conflicts_with = "udp_listen")]
    mic: bool,
    /// Bind a UDP socket and treat each datagram as raw f32-LE 16 kHz mono PCM.
    /// Example: `--udp-listen 0.0.0.0:9999`.
    #[arg(long)]
    udp_listen: Option<String>,
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
        { Device::new_metal(0).unwrap_or(Device::Cpu) }
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        { Device::new_cuda(0).unwrap_or(Device::Cpu) }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        { Device::Cpu }
    };
    let dtype = DType::F32;
    eprintln!("device: {:?}", device);

    let mel_cfg = MelConfig::nemotron_default();
    let mel = IncrementalMelExtractor::from_safetensors(&args.st, mel_cfg.clone())?;
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

    let mut source: Box<dyn AudioSource> = if let Some(p) = &args.audio {
        let samples = load_audio_mono_16k(p)?;
        // Feed in 320-sample (~20 ms) chunks to exercise the streaming
        // advance logic; the pipeline batches up internally.
        Box::new(FileChunkSource::new(samples, 320))
    } else if let Some(addr) = &args.udp_listen {
        let src = UdpSource::bind(addr, 320).await?;
        eprintln!("UDP listening on {}", src.local_addr()?);
        Box::new(src)
    } else if args.mic {
        #[cfg(feature = "mic")]
        {
            Box::new(MicSource::open_default(320)?)
        }
        #[cfg(not(feature = "mic"))]
        {
            anyhow::bail!("rebuild with --features mic to use microphone input");
        }
    } else {
        anyhow::bail!("specify --audio <file>, --mic, or --udp-listen <addr>");
    };

    eprintln!("listening... (Ctrl-C to stop)");
    std::io::stderr().flush().ok();

    // Index up to which we've already emitted text. We hold back any trailing
    // partial-word run so the next chunk's pieces can complete it without
    // splitting a word across two log lines.
    let mut emitted_idx: usize = 0;
    // If no new token has been produced for this long, the held-back tail
    // must be a complete word — flush it.
    let idle_flush = std::time::Duration::from_millis(600);
    let mut last_token_time = std::time::Instant::now();

    loop {
        match source.next_chunk().await? {
            None => break,
            Some(chunk) => {
                let is_final = chunk.is_final;
                pipe.push_audio(&chunk.samples);
                if is_final {
                    pipe.finish();
                }
                let prev_total = pipe.all_tokens.len();
                while let Some(_) = pipe.try_advance()? {}
                let n = pipe.all_tokens.len();
                let now = std::time::Instant::now();
                if n > prev_total {
                    last_token_time = now;
                }

                let idle_long_enough = now.duration_since(last_token_time) >= idle_flush;
                let upto = if is_final || idle_long_enough {
                    n
                } else {
                    last_word_initial(&tok, &pipe.all_tokens, emitted_idx, n)?
                        .unwrap_or(emitted_idx)
                };
                if upto > emitted_idx {
                    let prev = if emitted_idx == 0 {
                        String::new()
                    } else {
                        tok.detokenize(&pipe.all_tokens[..emitted_idx])?
                    };
                    let cur = tok.detokenize(&pipe.all_tokens[..upto])?;
                    let new_text = cur.strip_prefix(&prev).unwrap_or(&cur);
                    eprintln!("[chunk] {}", new_text);
                    std::io::stderr().flush().ok();
                    emitted_idx = upto;
                }
            }
        }
    }
    eprintln!();
    Ok(())
}

/// Find the highest index k in `[lo, hi)` such that `tokens[k]` starts a new
/// word (its decoded piece introduces a leading space, or k == 0). Returns
/// None if no such index exists in the range.
fn last_word_initial(
    tok: &Tokenizer,
    tokens: &[u32],
    lo: usize,
    hi: usize,
) -> Result<Option<usize>> {
    for k in (lo..hi).rev() {
        if k == 0 {
            return Ok(Some(0));
        }
        let before = tok.detokenize(&tokens[..k])?;
        let after = tok.detokenize(&tokens[..k + 1])?;
        if after.len() > before.len() && after.as_bytes()[before.len()] == b' ' {
            return Ok(Some(k));
        }
    }
    Ok(None)
}
