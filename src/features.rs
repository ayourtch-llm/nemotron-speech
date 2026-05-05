//! Log-mel spectrogram matching NeMo's `AudioToMelSpectrogramPreprocessor`
//! configured exactly as `nemotron-speech-streaming-en-0.6b` was trained:
//!
//! - 16 kHz mono f32 input
//! - preemphasis 0.97
//! - STFT: n_fft=512, win_length=400 (Hann), hop_length=160, center=True
//!   with reflection padding (matches torch.stft default)
//! - power spectrum (magnitude squared)
//! - 128 mel filters (Slaney-normalized; weights taken from the checkpoint)
//! - log(x + 2^-24)
//! - normalize: NA  (skipped, per model_config.yaml)
//!
//! The Hann window and mel filterbank are loaded from the converted
//! safetensors so they match the trained values to the bit.

use anyhow::{Context, Result, anyhow};
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub preemph: f32,
    pub log_zero_guard: f32,
}

impl MelConfig {
    pub fn nemotron_default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 128,
            preemph: 0.97,
            log_zero_guard: 2.0_f32.powi(-24),
        }
    }
}

/// Stateless mel extractor. Holds the FFT plan, the Hann window, and the
/// mel filterbank read from the model's checkpoint.
pub struct MelExtractor {
    cfg: MelConfig,
    window: Vec<f32>,           // (win_length,)
    mel_fb: Vec<f32>,           // row-major (n_mels, n_fft/2 + 1)
    fft: Arc<dyn RealToComplex<f32>>,
    in_buf: Vec<f32>,           // length n_fft, scratch for one frame
    out_buf: Vec<Complex<f32>>, // length n_fft/2 + 1, scratch for one frame
}

impl MelExtractor {
    /// Construct from raw window/filterbank tensors (already in the right
    /// layout: `window` of length `cfg.win_length`, `mel_fb` row-major
    /// `(cfg.n_mels, cfg.n_fft/2 + 1)`).
    pub fn new(cfg: MelConfig, window: Vec<f32>, mel_fb: Vec<f32>) -> Result<Self> {
        if window.len() != cfg.win_length {
            return Err(anyhow!(
                "window length mismatch: got {}, expected {}",
                window.len(),
                cfg.win_length
            ));
        }
        let n_bins = cfg.n_fft / 2 + 1;
        if mel_fb.len() != cfg.n_mels * n_bins {
            return Err(anyhow!(
                "mel_fb size mismatch: got {}, expected {}*{} = {}",
                mel_fb.len(),
                cfg.n_mels,
                n_bins,
                cfg.n_mels * n_bins
            ));
        }
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);
        let in_buf = vec![0.0f32; cfg.n_fft];
        let out_buf = vec![Complex::<f32>::new(0.0, 0.0); n_bins];
        Ok(Self {
            cfg,
            window,
            mel_fb,
            fft,
            in_buf,
            out_buf,
        })
    }

    /// Load window + mel_fb from a safetensors file (e.g. our converted
    /// checkpoint, which exposes `preproc.window` and `preproc.mel_fb`).
    pub fn from_safetensors<P: AsRef<Path>>(path: P, cfg: MelConfig) -> Result<Self> {
        use safetensors::SafeTensors;
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        let st = SafeTensors::deserialize(&bytes).context("safetensors parse")?;
        let window = read_f32_tensor(&st, "preproc.window")?;
        // mel_fb is shaped (1, n_mels, n_bins) in the checkpoint; flatten the leading 1.
        let mel_fb_raw = read_f32_tensor(&st, "preproc.mel_fb")?;
        Self::new(cfg, window, mel_fb_raw)
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }

    /// Number of output frames produced for a given audio length, matching
    /// torch.stft(center=True). Equivalent to ceil((n + 1) / hop) rounded
    /// per `floor((n + n_fft//2*2) / hop) + 1` minus pad/2... but easiest to
    /// just match torch by padding by `n_fft/2` reflectively on each side.
    pub fn n_frames(&self, n_samples: usize) -> usize {
        let pad = self.cfg.n_fft / 2;
        let padded = n_samples + 2 * pad;
        if padded < self.cfg.n_fft {
            0
        } else {
            (padded - self.cfg.n_fft) / self.cfg.hop_length + 1
        }
    }

    /// Compute the full log-mel spectrogram for an offline audio buffer.
    /// Returns a row-major `(n_mels, T)` flat array.
    pub fn forward(&mut self, audio: &[f32]) -> Vec<f32> {
        let n_mels = self.cfg.n_mels;
        let n_bins = self.cfg.n_fft / 2 + 1;
        let hop = self.cfg.hop_length;
        let win = self.cfg.win_length;
        let n_fft = self.cfg.n_fft;
        let pad = n_fft / 2;
        let win_offset = (n_fft - win) / 2; // 56 for n_fft=512, win=400

        // 1) Preemphasis: y[t] = x[t] - preemph * x[t-1], y[0] = x[0]
        let pre: Vec<f32> = if self.cfg.preemph != 0.0 {
            let p = self.cfg.preemph;
            let mut v = Vec::with_capacity(audio.len());
            if !audio.is_empty() {
                v.push(audio[0]);
                for i in 1..audio.len() {
                    v.push(audio[i] - p * audio[i - 1]);
                }
            }
            v
        } else {
            audio.to_vec()
        };

        // 2) Reflection-pad by n_fft/2 on each side (matches torch.stft center=True).
        // padded = pre[pad-1..0] reverse + pre + pre[..pre.len()-pad-1..pre.len()-1].rev
        let padded = reflect_pad(&pre, pad);
        let n_frames = if padded.len() < n_fft {
            0
        } else {
            (padded.len() - n_fft) / hop + 1
        };
        let mut out = vec![0.0f32; n_mels * n_frames];

        let log_eps = self.cfg.log_zero_guard;

        for t in 0..n_frames {
            let start = t * hop;
            // Apply window: frame[i] = padded[start+i] * window[i - win_offset]
            // but only in the central `win_length` samples. Outer samples are zero.
            // Note torch's center stft windows the n_fft-sized region using a
            // win_length-sized window centered. We follow the same convention.
            for s in self.in_buf.iter_mut() {
                *s = 0.0;
            }
            for i in 0..win {
                let src = start + win_offset + i;
                self.in_buf[win_offset + i] = padded[src] * self.window[i];
            }

            // FFT -> n_bins complex
            self.fft
                .process(&mut self.in_buf, &mut self.out_buf)
                .expect("fft");

            // power spectrum + mel filterbank multiply + log; fused for one frame.
            let row_off = t; // we store column-major below for simpler access
            for m in 0..n_mels {
                let row = &self.mel_fb[m * n_bins..(m + 1) * n_bins];
                let mut acc = 0.0f32;
                for b in 0..n_bins {
                    let c = self.out_buf[b];
                    let p = c.re * c.re + c.im * c.im;
                    acc += p * row[b];
                }
                // store as (n_mels, T) row-major: out[m * n_frames + t]
                out[m * n_frames + row_off] = (acc + log_eps).ln();
            }
        }
        out
    }
}

fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    // Left reflect: x[1..pad+1].rev()  (excludes the boundary sample, like numpy reflect)
    for i in 0..pad {
        let idx = (pad - i).min(n - 1);
        out.push(x[idx]);
    }
    out.extend_from_slice(x);
    // Right reflect: x[n-2 .. n-2-pad].rev()
    for i in 0..pad {
        let idx = if n >= 2 + i { n - 2 - i } else { 0 };
        out.push(x[idx]);
    }
    out
}

fn read_f32_tensor(st: &safetensors::SafeTensors, name: &str) -> Result<Vec<f32>> {
    let view = st
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    if !matches!(view.dtype(), safetensors::Dtype::F32) {
        return Err(anyhow!(
            "expected f32 for {name}, got {:?}",
            view.dtype()
        ));
    }
    let raw = view.data();
    if raw.len() % 4 != 0 {
        return Err(anyhow!("{name}: byte length not multiple of 4"));
    }
    let n = raw.len() / 4;
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < raw.len() {
        let bytes = [raw[i], raw[i + 1], raw[i + 2], raw[i + 3]];
        out.push(f32::from_le_bytes(bytes));
        i += 4;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_pad_basic() {
        // numpy reflect pad: pad=2, [1,2,3,4,5] -> [3,2,1,2,3,4,5,4,3]
        let p = reflect_pad(&[1.0, 2.0, 3.0, 4.0, 5.0], 2);
        assert_eq!(p, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }
}
