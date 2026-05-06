//! Offline AEC validator. Takes a mic WAV (with echo) and a time-aligned
//! reference WAV (the speaker signal), runs the kernel frame-by-frame,
//! and writes a cleaned 16 kHz mono 16-bit PCM WAV. The cleaned WAV can
//! then be transcribed (`transcribe --audio out.wav`) to confirm the
//! echo's text is gone (or that user words survive AEC when both are
//! present).
//!
//! Build: `cargo build --release --bin aec_check`
//! Run:   `./target/release/aec_check --mic mic.wav --reference ref.wav --out cleaned.wav`
//!
//! Both WAVs are treated as time-aligned: the reference samples at index
//! `i` correspond (modulo the echo's propagation delay) to the mic samples
//! at index `i`. In the live pipeline this alignment is approximate and
//! the kernel's cross-correlation search handles small skew; for offline
//! validation, generate the WAVs from the same clock.

use anyhow::{Context, Result};
use clap::Parser;
use nemotron_speech::aec::{AecKernel, ReferenceHistory, SpectralSubtractionAec};
use nemotron_speech::audio::load_audio_mono_16k;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Mic WAV (with echo). Decoded to 16 kHz mono f32.
    #[arg(long)]
    mic: PathBuf,
    /// Reference WAV (the speaker signal). Decoded to 16 kHz mono f32.
    #[arg(long)]
    reference: PathBuf,
    /// Output cleaned WAV (16 kHz mono 16-bit PCM).
    #[arg(long)]
    out: PathBuf,
    /// Frame size in samples. 320 = 20 ms at 16 kHz, matches the live
    /// pipeline's chunking.
    #[arg(long, default_value_t = 320)]
    frame: usize,
    /// Reference history capacity in samples. 48000 = 3 s at 16 kHz.
    #[arg(long, default_value_t = 48_000)]
    history: usize,
    /// Print per-frame AEC stats (delay, confidence, gain).
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    let mic = load_audio_mono_16k(&args.mic)
        .with_context(|| format!("loading mic {}", args.mic.display()))?;
    let reference = load_audio_mono_16k(&args.reference)
        .with_context(|| format!("loading reference {}", args.reference.display()))?;
    eprintln!(
        "loaded {} mic samples, {} reference samples",
        mic.len(),
        reference.len()
    );

    let mut history = ReferenceHistory::new(args.history);
    let mut kernel = SpectralSubtractionAec::new();

    let frame = args.frame;
    let mut cleaned: Vec<f32> = Vec::with_capacity(mic.len());

    let mut pos = 0;
    let mut ref_pos = 0;
    let mut mic_energy = 0f32;
    let mut clean_energy = 0f32;
    while pos + frame <= mic.len() {
        // Push reference samples up to the same wall-clock as this mic
        // frame. In the live pipeline the producer task does this for
        // us; offline we synchronize manually.
        let ref_target = (pos + frame).min(reference.len());
        if ref_target > ref_pos {
            history.push(&reference[ref_pos..ref_target]);
            ref_pos = ref_target;
        }

        let snap = history.snapshot();
        let out_frame = kernel.process(&mic[pos..pos + frame], &snap);

        for i in 0..frame {
            mic_energy += mic[pos + i] * mic[pos + i];
            clean_energy += out_frame[i] * out_frame[i];
        }
        if args.verbose {
            if let Some(s) = kernel.last_frame_stats() {
                eprintln!(
                    "pos={pos:>6} best_d={:>4} conf={:.3} gain={:+.3} mic_e={:.4} ref_e={:.4} delay_est={:.1}",
                    s.best_d,
                    s.confidence,
                    s.gain,
                    s.mic_energy,
                    s.ref_energy,
                    kernel.delay_estimate(),
                );
            }
        }
        cleaned.extend(out_frame);
        pos += frame;
    }

    // Write 16-bit PCM WAV.
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&args.out, spec)
        .with_context(|| format!("creating {}", args.out.display()))?;
    for &s in &cleaned {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;

    let suppression_db = if clean_energy > 0.0 {
        10.0 * (mic_energy / clean_energy).log10()
    } else {
        f32::INFINITY
    };
    eprintln!(
        "wrote {} ({} samples). final delay estimate: {:.1} samples ({:.1} ms). \
         residual suppression: {:.1} dB",
        args.out.display(),
        cleaned.len(),
        kernel.delay_estimate(),
        kernel.delay_estimate() / 16.0,
        suppression_db,
    );
    Ok(())
}
