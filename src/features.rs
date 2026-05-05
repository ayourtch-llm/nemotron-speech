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

/// Stateful streaming variant of `MelExtractor`.
///
/// Audio is pushed in arbitrary chunks; emittable mel frames become available
/// as enough samples are buffered to compute them WITHOUT right-side
/// reflection. Calling `finish()` then unlocks the final 1–2 frames using
/// right-side reflection (matching the offline torch.stft `center=True` tail).
///
/// Pre-emphasis is run incrementally with a one-sample history. The internal
/// pre-emphasized sample buffer is pruned as frames advance — only the
/// minimum needed for the next frame's window is retained.
///
/// Output layout (`mel_buffer`) is `(n_mels, T)` row-major, matching the
/// offline `forward()` so a `StreamingPipeline` can drop it into a tensor
/// without transposing.
pub struct IncrementalMelExtractor {
    cfg: MelConfig,
    window: Vec<f32>,
    mel_fb: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    in_buf: Vec<f32>,
    out_buf: Vec<Complex<f32>>,

    /// Pre-emphasized samples, with `pre[0]` corresponding to original-stream
    /// index `pre_offset`. Older samples are pruned once no future frame needs
    /// them.
    pre: Vec<f32>,
    pre_offset: usize,
    /// Last raw sample seen, for preemphasis continuation across pushes.
    last_raw: Option<f32>,
    /// Total raw samples ever pushed (across all calls).
    raw_count: usize,
    /// Number of mel frames already produced and stored in `mel_rows`.
    frames_emitted: usize,
    finished: bool,

    /// One Vec per mel bin, each of length `frames_emitted`. Storing per-row
    /// makes appending a frame an `n_mels`-element push, and flattening to
    /// `(n_mels, T)` row-major is a sequence of `extend_from_slice` calls.
    mel_rows: Vec<Vec<f32>>,
}

impl IncrementalMelExtractor {
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
        let mel_rows = vec![Vec::new(); cfg.n_mels];
        Ok(Self {
            in_buf: vec![0.0f32; cfg.n_fft],
            out_buf: vec![Complex::<f32>::new(0.0, 0.0); n_bins],
            cfg,
            window,
            mel_fb,
            fft,
            pre: Vec::new(),
            pre_offset: 0,
            last_raw: None,
            raw_count: 0,
            frames_emitted: 0,
            finished: false,
            mel_rows,
        })
    }

    pub fn from_safetensors<P: AsRef<Path>>(path: P, cfg: MelConfig) -> Result<Self> {
        use safetensors::SafeTensors;
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        let st = SafeTensors::deserialize(&bytes).context("safetensors parse")?;
        let window = read_f32_tensor(&st, "preproc.window")?;
        let mel_fb_raw = read_f32_tensor(&st, "preproc.mel_fb")?;
        Self::new(cfg, window, mel_fb_raw)
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }

    pub fn n_frames_emitted(&self) -> usize {
        self.frames_emitted
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Append more raw audio. Pre-emphasis runs incrementally using
    /// `last_raw` for continuity; the very first sample of the entire stream
    /// follows the offline convention `pre[0] = audio[0]`.
    pub fn push_audio(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let p = self.cfg.preemph;
        let mut last = self.last_raw;
        if p == 0.0 {
            self.pre.extend_from_slice(samples);
        } else {
            for &s in samples {
                let pre = match last {
                    Some(prev) => s - p * prev,
                    None => s,
                };
                self.pre.push(pre);
                last = Some(s);
            }
        }
        self.last_raw = if p == 0.0 {
            samples.last().copied().or(self.last_raw)
        } else {
            last
        };
        self.raw_count += samples.len();
        self.compute_available_frames();
    }

    /// Mark the input stream as ended — unlocks the trailing frames that need
    /// right-side reflection.
    pub fn finish(&mut self) {
        if !self.finished {
            self.finished = true;
            self.compute_available_frames();
        }
    }

    /// Number of mel frames computable now. Matches the offline `n_frames()`
    /// once `finish()` has been called.
    pub fn available_frames(&self) -> usize {
        let hop = self.cfg.hop_length;
        let win = self.cfg.win_length;
        let half_win = win / 2;
        if self.finished {
            if self.raw_count == 0 {
                0
            } else {
                self.raw_count / hop + 1
            }
        } else if self.raw_count >= half_win {
            (self.raw_count - half_win) / hop + 1
        } else {
            0
        }
    }

    /// Read the cumulative mel buffer in `(n_mels, T)` row-major layout.
    /// Caller can build a Tensor with shape `(1, n_mels, T)` directly.
    pub fn mel_buffer(&self) -> Vec<f32> {
        let n_mels = self.cfg.n_mels;
        let t = self.frames_emitted;
        let mut out = Vec::with_capacity(n_mels * t);
        for m in 0..n_mels {
            out.extend_from_slice(&self.mel_rows[m]);
        }
        out
    }

    /// Return mel frames `[start, frames_emitted)` in `(n_mels, K)` row-major
    /// layout. Useful for feeding only newly-arrived frames to a streaming
    /// subsample stack.
    pub fn mel_buffer_since(&self, start: usize) -> Vec<f32> {
        let n_mels = self.cfg.n_mels;
        let end = self.frames_emitted;
        let k = end.saturating_sub(start);
        let mut out = Vec::with_capacity(n_mels * k);
        for m in 0..n_mels {
            out.extend_from_slice(&self.mel_rows[m][start..end]);
        }
        out
    }

    fn compute_available_frames(&mut self) {
        let avail = self.available_frames();
        if avail <= self.frames_emitted {
            return;
        }
        let n_mels = self.cfg.n_mels;
        let n_bins = self.cfg.n_fft / 2 + 1;
        let hop = self.cfg.hop_length;
        let win = self.cfg.win_length;
        let n_fft = self.cfg.n_fft;
        let win_offset = (n_fft - win) / 2;
        let half_win = win / 2;
        let log_eps = self.cfg.log_zero_guard;

        for t in self.frames_emitted..avail {
            let start = t * hop;
            for s in self.in_buf.iter_mut() {
                *s = 0.0;
            }
            for i in 0..win {
                let orig_idx = (start as isize) - (half_win as isize) + (i as isize);
                let val = self.fetch(orig_idx);
                self.in_buf[win_offset + i] = val * self.window[i];
            }
            self.fft
                .process(&mut self.in_buf, &mut self.out_buf)
                .expect("fft");
            for m in 0..n_mels {
                let row = &self.mel_fb[m * n_bins..(m + 1) * n_bins];
                let mut acc = 0.0f32;
                for b in 0..n_bins {
                    let c = self.out_buf[b];
                    let p = c.re * c.re + c.im * c.im;
                    acc += p * row[b];
                }
                self.mel_rows[m].push((acc + log_eps).ln());
            }
        }

        self.frames_emitted = avail;

        // Prune: the next frame to compute is `frames_emitted`, which needs
        // original-index `frames_emitted*hop - half_win` and onward. Drop pre
        // samples below that.
        let next_needed = (self.frames_emitted as isize) * (hop as isize) - (half_win as isize);
        if next_needed > self.pre_offset as isize {
            let drop = (next_needed as usize) - self.pre_offset;
            let drop = drop.min(self.pre.len());
            self.pre.drain(0..drop);
            self.pre_offset += drop;
        }
    }

    /// Fetch pre-emphasized sample at original-stream index `k`, applying
    /// torch-style symmetric reflection (excluding the boundary sample) when
    /// `k` is outside `[0, raw_count)`.
    fn fetch(&self, k: isize) -> f32 {
        let n = self.raw_count as isize;
        let k = if k < 0 {
            -k
        } else if k >= n {
            // Right-reflection only valid once finished; in streaming mode we
            // shouldn't be asked for samples beyond the end.
            debug_assert!(self.finished, "right-reflect fetch in streaming mode");
            2 * n - 2 - k
        } else {
            k
        };
        let k = k.clamp(0, (n - 1).max(0));
        let off = k as usize;
        debug_assert!(off >= self.pre_offset, "pruned sample fetch");
        self.pre[off - self.pre_offset]
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

    /// Build a deterministic Hann-like window and a tiny synthetic mel
    /// filterbank for tests. Real precision matters: we want bit-equivalence
    /// across offline vs incremental on the same input, so use the SAME
    /// window/filterbank for both.
    fn synthetic_pieces() -> (MelConfig, Vec<f32>, Vec<f32>) {
        let cfg = MelConfig::nemotron_default();
        let n_bins = cfg.n_fft / 2 + 1;
        // Hann window of length win_length
        let win: Vec<f32> = (0..cfg.win_length)
            .map(|i| {
                let x = (i as f32) / ((cfg.win_length - 1) as f32);
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
            })
            .collect();
        // A simple smooth filterbank: gaussian bumps per mel bin.
        let mut mel_fb = vec![0.0f32; cfg.n_mels * n_bins];
        for m in 0..cfg.n_mels {
            let center = (m as f32) * (n_bins as f32) / (cfg.n_mels as f32);
            let width = 4.0;
            for b in 0..n_bins {
                let z = ((b as f32) - center) / width;
                mel_fb[m * n_bins + b] = (-0.5 * z * z).exp();
            }
        }
        (cfg, win, mel_fb)
    }

    fn synthetic_audio(n_samples: usize) -> Vec<f32> {
        // Mix of two sinusoids — enough spectral content to exercise the mel.
        (0..n_samples)
            .map(|i| {
                let t = (i as f32) / 16_000.0;
                0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()
            })
            .collect()
    }

    /// Compare two `(n_mels, T)` row-major buffers; return (max_abs, mean_abs).
    fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
        assert_eq!(a.len(), b.len(), "shape mismatch");
        let mut max = 0.0f32;
        let mut sum = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > max {
                max = d;
            }
            sum += d as f64;
        }
        (max, (sum / (a.len() as f64)) as f32)
    }

    #[test]
    fn incremental_matches_offline_after_finish() {
        let (cfg, win, mel_fb) = synthetic_pieces();
        let audio = synthetic_audio(16_000); // 1 second
        let mut offline = MelExtractor::new(cfg.clone(), win.clone(), mel_fb.clone()).unwrap();
        let off = offline.forward(&audio);

        // Stream the audio in 320-sample (20 ms) chunks.
        let mut inc = IncrementalMelExtractor::new(cfg.clone(), win, mel_fb).unwrap();
        for chunk in audio.chunks(320) {
            inc.push_audio(chunk);
        }
        inc.finish();
        let inc_out = inc.mel_buffer();

        assert_eq!(inc.n_frames_emitted() * cfg.n_mels, inc_out.len());
        assert_eq!(off.len(), inc_out.len(), "frame count differs");
        let (max, mean) = diff_stats(&off, &inc_out);
        // Same arithmetic operations, same order, identical floats expected
        // up to FFT scratch reuse. Allow a very tight bound.
        assert!(max < 1e-5, "max abs diff too large: {max} (mean {mean})");
    }

    #[test]
    fn incremental_streaming_lags_offline_by_one_frame_min() {
        // Mid-stream (no `finish()`) the incremental output should lack
        // exactly the frames that depend on right-side reflection — at
        // n_fft/(2*hop) ≈ 1.6 trailing frames.
        let (cfg, win, mel_fb) = synthetic_pieces();
        let audio = synthetic_audio(8_000);
        let mut offline = MelExtractor::new(cfg.clone(), win.clone(), mel_fb.clone()).unwrap();
        let off = offline.forward(&audio);
        let off_t = off.len() / cfg.n_mels;

        let mut inc = IncrementalMelExtractor::new(cfg, win, mel_fb).unwrap();
        for chunk in audio.chunks(160) {
            inc.push_audio(chunk);
        }
        let inc_t = inc.n_frames_emitted();
        let lag = off_t - inc_t;
        assert!(lag == 1 || lag == 2, "expected 1–2 frame lag, got {lag} (off={off_t} inc={inc_t})");
    }
}
