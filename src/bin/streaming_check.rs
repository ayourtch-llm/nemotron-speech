//! Validate cache-aware chunked encoder forward against the reference path.
//!
//! Plan:
//!   1. Load audio, compute mel, run the subsample stack offline.
//!   2. (Path A, reference) run the 24-layer encoder over the full
//!      subsampled sequence with the chunked-limited mask applied.
//!   3. (Path B, streaming) split the subsampled sequence into chunks of
//!      `chunk_size_enc_frames` and call `forward_layers_chunked` once per
//!      chunk with a sliding KV cache + conv state cache.
//!   4. Concatenate the per-chunk outputs and compare against path A.
//!
//! For the 5-second test clip the full attention path equals the chunked
//! path (only 5 chunks, all within the left-context limit), so both paths
//! also equal the existing offline transcription.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use nemotron_speech::audio;
use nemotron_speech::features::{MelConfig, MelExtractor};
use nemotron_speech::model::ModelConfig;
use nemotron_speech::model::encoder::{EncoderCache, FastConformerEncoder};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    audio: PathBuf,
    #[arg(long)]
    st: PathBuf,
    #[arg(long, default_value_t = 5e-3)]
    atol: f32,
    /// Number of chunks to fuse into one `forward_layers_chunked` call.
    /// 1 = original per-chunk path. >1 exercises the block-causal batched
    /// mask; output must still match the reference within `atol`.
    #[arg(long, default_value_t = 1)]
    batch: usize,
    /// Run on Metal instead of CPU (requires the `metal` feature). Lets us
    /// isolate the conformer-encoder cost on the GPU from the rest of the
    /// live pipeline.
    #[arg(long, default_value_t = false)]
    metal: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    let device = if args.metal {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).context("opening Metal device")?
        }
        #[cfg(not(feature = "metal"))]
        {
            bail!("rebuild with --features metal to use --metal");
        }
    } else {
        Device::Cpu
    };
    let dtype = DType::F32;
    println!("device: {:?}", device);

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

    // Reference: full-utt encoder with chunked-limited attention mask.
    let t0 = std::time::Instant::now();
    let ref_out = encoder
        .forward_full(&mel_t, true)
        .map_err(|e| anyhow::anyhow!("reference forward: {e:#}"))?;
    let t_ref = t0.elapsed();
    let (_b, t_total, d) = ref_out.dims3()?;
    println!(
        "reference (full-utt + chunked mask): shape {:?} in {:?}",
        ref_out.dims(),
        t_ref
    );

    // Streaming: subsample the full mel, then split the encoded sequence
    // into chunks and call forward_layers_chunked per chunk with caches.
    let subsampled = encoder
        .subsample
        .forward(&mel_t)
        .map_err(|e| anyhow::anyhow!("subsample: {e:#}"))?;
    let chunk_size = cfg.chunk_size_enc_frames();
    let batch = args.batch.max(1);
    let step = chunk_size * batch;
    let n_steps = (t_total + step - 1) / step;
    let mut cache = EncoderCache::empty(cfg.n_layers);
    let mut chunk_outs: Vec<Tensor> = Vec::with_capacity(n_steps);

    let t0 = std::time::Instant::now();
    for c in 0..n_steps {
        let start = c * step;
        let len = step.min(t_total - start);
        let chunk = subsampled.narrow(1, start, len)?.contiguous()?;
        let out = encoder
            .forward_layers_chunked(&chunk, &mut cache)
            .map_err(|e| anyhow::anyhow!("step {} forward: {e:#}", c))?;
        chunk_outs.push(out);
    }
    let t_stream = t0.elapsed();
    let stream_out = Tensor::cat(&chunk_outs.iter().collect::<Vec<_>>(), 1)?;
    println!(
        "streaming ({} passes of up to {} frames, batch={}): shape {:?} in {:?}",
        n_steps,
        step,
        batch,
        stream_out.dims(),
        t_stream
    );

    // Diff
    let ref_v: Vec<f32> = ref_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let stream_v: Vec<f32> = stream_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    if ref_v.len() != stream_v.len() {
        bail!(
            "size mismatch: ref {} vs stream {}",
            ref_v.len(),
            stream_v.len()
        );
    }
    let mut max_abs = 0.0f32;
    let mut argmax = 0usize;
    let mut sum = 0.0f64;
    for i in 0..ref_v.len() {
        let d = (ref_v[i] - stream_v[i]).abs();
        sum += d as f64;
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
    }
    let mean = sum / ref_v.len() as f64;
    let frame_idx = argmax / d;
    let ch_idx = argmax % d;
    println!(
        "max abs diff {:.3e} (at frame={}, ch={}; ref={:.4} stream={:.4}), mean {:.3e}",
        max_abs, frame_idx, ch_idx, ref_v[argmax], stream_v[argmax], mean
    );
    if max_abs > args.atol {
        bail!(
            "FAIL: max diff {:.3e} exceeds tol {:.3e}",
            max_abs,
            args.atol
        );
    }
    println!("OK");
    Ok(())
}
