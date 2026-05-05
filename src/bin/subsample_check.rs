//! Validate the streaming subsample stack against the offline path.
//!
//! Loads the model, computes mel features, then:
//!   - Path A: runs `DwStridingSubsampling::forward(full_mel)` once.
//!   - Path B: feeds the mel in slices through `forward_incremental` with
//!     a fresh `SubsampleStreamingState`; calls `finished=true` at the end
//!     to flush trailing tentative frames.
//! Both paths should produce identical `(1, T, d_model)` output.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use nemotron_speech::audio;
use nemotron_speech::features::{MelConfig, MelExtractor};
use nemotron_speech::model::ModelConfig;
use nemotron_speech::model::encoder::{FastConformerEncoder, SubsampleStreamingState};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    audio: PathBuf,
    #[arg(long)]
    st: PathBuf,
    /// Mel-frame slice size for the incremental path.
    #[arg(long, default_value_t = 32)]
    slice: usize,
    #[arg(long, default_value_t = 1e-5)]
    atol: f32,
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

    let audio_v = audio::load_audio_mono_16k(&args.audio)?;
    let mel_cfg = MelConfig::nemotron_default();
    let mut mel = MelExtractor::from_safetensors(&args.st, mel_cfg.clone())?;
    let log_mel = mel.forward(&audio_v);
    let n_frames = log_mel.len() / mel_cfg.n_mels;
    let mel_t = Tensor::from_vec(log_mel.clone(), (1, mel_cfg.n_mels, n_frames), &device)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.st.clone()], dtype, &device)
            .context("loading safetensors")?
    };
    let cfg = ModelConfig::nemotron_06b();
    let encoder = FastConformerEncoder::new(vb.pp("encoder"), cfg.clone())
        .map_err(|e| anyhow::anyhow!("encoder: {e:#}"))?;

    // Path A: offline.
    let off = encoder
        .subsample
        .forward(&mel_t)
        .map_err(|e| anyhow::anyhow!("offline subsample: {e:#}"))?;
    let off_dims = off.dims().to_vec();
    println!("offline: {:?}", off_dims);

    // Path B: incremental, mel fed in fixed-size slices.
    let mut state = SubsampleStreamingState::empty();
    let n_mels = mel_cfg.n_mels;
    let mut consumed = 0;
    while consumed < n_frames {
        let take = args.slice.min(n_frames - consumed);
        // Build (1, n_mels, take) tensor for this slice. Source layout is (n_mels, T) row-major.
        let mut slab = Vec::with_capacity(n_mels * take);
        for m in 0..n_mels {
            let row_start = m * n_frames + consumed;
            slab.extend_from_slice(&log_mel[row_start..row_start + take]);
        }
        let slice_t = Tensor::from_vec(slab, (1, n_mels, take), &device)?;
        let last = consumed + take == n_frames;
        encoder
            .subsample
            .forward_incremental(&slice_t, &mut state, &device, dtype, last)
            .map_err(|e| anyhow::anyhow!("incremental subsample: {e:#}"))?;
        consumed += take;
    }
    let inc = state
        .output
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no incremental output produced"))?;
    let inc_dims = inc.dims().to_vec();
    println!("incremental ({} mel-frame slices): {:?}", args.slice, inc_dims);

    if off_dims != inc_dims {
        bail!("shape mismatch: offline {:?} vs incremental {:?}", off_dims, inc_dims);
    }
    let off_v: Vec<f32> = off.flatten_all()?.to_vec1()?;
    let inc_v: Vec<f32> = inc.flatten_all()?.to_vec1()?;
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    let mut argmax = 0usize;
    for i in 0..off_v.len() {
        let d = (off_v[i] - inc_v[i]).abs();
        sum += d as f64;
        if d > max {
            max = d;
            argmax = i;
        }
    }
    let mean = sum / off_v.len() as f64;
    println!(
        "max abs diff {:.3e} (at idx {}; offline={:.4} incremental={:.4}), mean {:.3e}",
        max, argmax, off_v[argmax], inc_v[argmax], mean
    );
    if max > args.atol {
        bail!("FAIL: max diff {:.3e} exceeds tol {:.3e}", max, args.atol);
    }
    println!("OK");
    Ok(())
}
