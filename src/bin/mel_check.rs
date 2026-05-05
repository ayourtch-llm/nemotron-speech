//! Compute the log-mel spectrogram of a WAV file and compare it against a
//! reference produced by `tools/reference_mel.py`. Used as the offline
//! correctness gate for the feature extractor before we wire it into the
//! encoder.
//!
//! Usage:
//!     cargo run --bin mel_check -- \
//!         --wav tmp/small-test.wav \
//!         --st  models/nemotron-speech-streaming-en-0.6b.safetensors \
//!         --ref tmp/reference_mel.bin

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use nemotron_speech::audio;
use nemotron_speech::features::{MelConfig, MelExtractor};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    wav: PathBuf,
    #[arg(long)]
    st: PathBuf,
    #[arg(long = "ref")]
    reference: PathBuf,
    /// Maximum allowed absolute deviation from the Python reference (per element).
    #[arg(long, default_value_t = 1e-3)]
    atol: f32,
}

fn read_reference_bin(path: &std::path::Path) -> Result<(usize, usize, Vec<f32>)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 8 {
        bail!("reference too short");
    }
    let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let cols = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let expected = 8 + rows * cols * 4;
    if bytes.len() != expected {
        bail!(
            "ref size mismatch: header {}x{} expects {} bytes, got {}",
            rows,
            cols,
            expected,
            bytes.len()
        );
    }
    let mut data = Vec::with_capacity(rows * cols);
    let mut i = 8;
    while i < bytes.len() {
        let v = f32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
        data.push(v);
        i += 4;
    }
    Ok((rows, cols, data))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    let (audio, sr) = audio::load_pcm16_mono_wav(&args.wav)?;
    if sr != 16_000 {
        return Err(anyhow!("expected 16 kHz wav, got {sr}"));
    }
    println!(
        "audio: {} samples ({:.3}s) at {} Hz",
        audio.len(),
        audio.len() as f32 / sr as f32,
        sr
    );

    let cfg = MelConfig::nemotron_default();
    let mut mel = MelExtractor::from_safetensors(&args.st, cfg.clone())?;
    let n_frames = mel.n_frames(audio.len());
    println!("expecting {} mel frames", n_frames);

    let t0 = std::time::Instant::now();
    let log_mel = mel.forward(&audio);
    let took = t0.elapsed();
    let n_mels = cfg.n_mels;
    println!(
        "mel: {}x{} computed in {:?} ({:.1} kHz frames/sec)",
        n_mels,
        n_frames,
        took,
        (n_frames as f64 / took.as_secs_f64()) / 1000.0
    );

    let (rrows, rcols, ref_data) = read_reference_bin(&args.reference)?;
    if rrows != n_mels || rcols != n_frames {
        bail!(
            "reference shape {}x{} differs from rust shape {}x{}",
            rrows,
            rcols,
            n_mels,
            n_frames
        );
    }

    // Compare element-wise. log_mel is row-major (n_mels, T) which matches the
    // reference layout written by reference_mel.py.
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut argmax = (0usize, 0usize);
    for m in 0..n_mels {
        for t in 0..n_frames {
            let i = m * n_frames + t;
            let d = (log_mel[i] - ref_data[i]).abs();
            sum_abs += d as f64;
            if d > max_abs {
                max_abs = d;
                argmax = (m, t);
            }
        }
    }
    let mean_abs = sum_abs / (n_mels * n_frames) as f64;
    let (m, t) = argmax;
    println!(
        "abs diff: max={:.3e} (at mel={}, t={}; rust={:.4} ref={:.4}), mean={:.3e}",
        max_abs,
        m,
        t,
        log_mel[m * n_frames + t],
        ref_data[m * n_frames + t],
        mean_abs
    );

    if max_abs > args.atol {
        eprintln!("FAIL: max diff {:.3e} exceeds tolerance {:.3e}", max_abs, args.atol);
        std::process::exit(1);
    }
    println!("OK: mel matches reference within {:.3e}", args.atol);
    Ok(())
}
