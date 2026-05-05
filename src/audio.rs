//! Audio I/O. Decodes WAV and AAC/M4A via symphonia into 16 kHz mono f32.
//!
//! The streaming side of the project will consume audio from many sources
//! (mic, UDP, file). This module currently handles only file decode + resample
//! into the canonical 16 kHz mono f32 representation that the rest of the
//! pipeline assumes.

use anyhow::{Context, Result, anyhow};
use std::path::Path;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const TARGET_SR: u32 = 16_000;

/// Decode a file into 16 kHz mono f32 samples in [-1.0, 1.0].
///
/// If the file is not already 16 kHz mono, this routine averages channels
/// and does a simple linear-interpolation resample. Linear is fine for ASR
/// preprocessing on speech signals; we'll swap in something better later if
/// needed.
pub fn load_audio_mono_16k<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path.as_ref())
        .with_context(|| format!("opening {}", path.as_ref().display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.as_ref().extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("probe failed")?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("no default track"))?;
    let codec_params = track.codec_params.clone();
    let track_id = track.id;
    let src_sr = codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("missing sample rate"))?;
    let n_channels = codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .context("codec init")?;

    let mut mono: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e).context("packet read"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => append_mono(&buf, n_channels, &mut mono),
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decode")?,
        }
    }

    if src_sr == TARGET_SR {
        Ok(mono)
    } else {
        Ok(linear_resample(&mono, src_sr, TARGET_SR))
    }
}

fn append_mono(buf: &AudioBufferRef<'_>, _channels: usize, out: &mut Vec<f32>) {
    // Convert to f32 mono by averaging channels.
    match buf {
        AudioBufferRef::F32(b) => mix_to_mono_f32(b, out),
        AudioBufferRef::F64(b) => {
            let n = b.frames();
            let nc = b.spec().channels.count();
            out.reserve(n);
            for i in 0..n {
                let mut s = 0.0f64;
                for c in 0..nc {
                    s += b.chan(c)[i];
                }
                out.push((s / nc as f64) as f32);
            }
        }
        AudioBufferRef::S16(b) => {
            let n = b.frames();
            let nc = b.spec().channels.count();
            out.reserve(n);
            for i in 0..n {
                let mut s = 0.0f32;
                for c in 0..nc {
                    s += b.chan(c)[i] as f32;
                }
                out.push((s / nc as f32) / 32768.0);
            }
        }
        AudioBufferRef::S32(b) => {
            let n = b.frames();
            let nc = b.spec().channels.count();
            out.reserve(n);
            for i in 0..n {
                let mut s = 0.0f64;
                for c in 0..nc {
                    s += b.chan(c)[i] as f64;
                }
                out.push((s / nc as f64 / 2147483648.0) as f32);
            }
        }
        AudioBufferRef::U8(b) => {
            let n = b.frames();
            let nc = b.spec().channels.count();
            out.reserve(n);
            for i in 0..n {
                let mut s = 0.0f32;
                for c in 0..nc {
                    s += (b.chan(c)[i] as f32 - 128.0) / 128.0;
                }
                out.push(s / nc as f32);
            }
        }
        _ => panic!("unsupported sample format from symphonia"),
    }
}

fn mix_to_mono_f32(b: &symphonia::core::audio::AudioBuffer<f32>, out: &mut Vec<f32>) {
    let n = b.frames();
    let nc = b.spec().channels.count();
    out.reserve(n);
    if nc == 1 {
        out.extend_from_slice(b.chan(0));
        return;
    }
    for i in 0..n {
        let mut s = 0.0f32;
        for c in 0..nc {
            s += b.chan(c)[i];
        }
        out.push(s / nc as f32);
    }
}

fn linear_resample(x: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || x.is_empty() {
        return x.to_vec();
    }
    let ratio = dst_sr as f64 / src_sr as f64;
    let out_len = ((x.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    let step = src_sr as f64 / dst_sr as f64;
    for i in 0..out_len {
        let src_idx = i as f64 * step;
        let lo = src_idx.floor() as usize;
        let frac = (src_idx - lo as f64) as f32;
        let s0 = x[lo.min(x.len() - 1)];
        let s1 = x[(lo + 1).min(x.len() - 1)];
        out.push(s0 + (s1 - s0) * frac);
    }
    out
}

/// Read a 16-bit mono PCM WAV; faster path used by tests when we know the
/// format up-front (avoids symphonia init overhead).
pub fn load_pcm16_mono_wav<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path.as_ref())
        .with_context(|| format!("opening {}", path.as_ref().display()))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(anyhow!("expected mono wav, got {} channels", spec.channels));
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(anyhow!(
            "expected 16-bit PCM; got {:?} {}-bit",
            spec.sample_format,
            spec.bits_per_sample
        ));
    }
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<_, _>>()?;
    Ok((samples, spec.sample_rate))
}
