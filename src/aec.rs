//! Acoustic Echo Cancellation.
//!
//! Subtracts the speaker's reference signal (TTS audio that's about to be
//! played) from the microphone signal before ASR transcribes it. With AEC
//! on, the agent's own voice doesn't reach the LLM as a "user" turn — the
//! mic-loop feedback that produced the 12-minute, 179-WAV elephant-fact
//! cascade is broken at the audio layer instead of patched at the text
//! layer (where token-set Jaccard misses fragmented echoes).
//!
//! Spec: `kokoro-tts/docs/specs/m3-5-echo-cancellation.md`.
//!
//! v1 is time-domain: per mic frame, find the propagation delay via
//! cross-correlation against the reference history, compute a scalar
//! gain by least squares, and subtract. The kernel sits behind a trait
//! so a future NLMS / RLS adaptive filter (option c in the spec) can be
//! dropped in without touching the integration.
//!
//! Threading: `ReferenceHistory` is a plain ring buffer; a UDP listener
//! task pushes into it, and the main transcribe loop reads a snapshot
//! before each mic frame. Lock contention is negligible (~50 Hz on each
//! side, no work held under the lock besides memcpy).

use std::collections::VecDeque;

/// Sample rate everything in this module assumes (matches mic + reference).
pub const SR: u32 = 16_000;

/// Trait for an echo-cancellation algorithm. Implementations are stateful
/// across frames (e.g. they smooth a delay estimate or carry filter taps).
pub trait AecKernel: Send {
    /// Process one mic frame against the most recent reference samples.
    /// `ref_history` is oldest-first; the kernel searches for the
    /// propagation delay inside it. Returns cleaned samples of the same
    /// length as `mic_frame`. If the reference is silent, or there isn't
    /// enough history yet, returns the mic frame unchanged.
    fn process(&mut self, mic_frame: &[f32], ref_history: &[f32]) -> Vec<f32>;
}

/// v1 AEC: time-domain cross-correlation alignment + least-squares scalar
/// gain + subtraction. Robust to small clock skew (the search re-runs each
/// frame), cheap (a few hundred kFLOPs per 20 ms frame), and stateless
/// enough to fail open on silence.
///
/// Despite the name, this isn't *spectral* subtraction — that would need an
/// FFT and a magnitude-domain subtract. We kept the "spectral" name because
/// the spec uses it; the time-domain variant is what shipped (per spec §3,
/// "time-domain version is fine for v1").
///
/// The cross-correlation search and the LSQ gain estimate run over a
/// kernel-internal mic-history buffer (default 2048 samples ≈ 128 ms),
/// not just the current frame. Speech has strong autocorrelation at the
/// pitch period (~5–15 ms), which produces spurious peaks if the window
/// is short — a frame-sized search would lock onto a pitch multiple
/// rather than the true echo delay. The longer window averages those
/// out. The subtraction itself still applies frame-by-frame.
pub struct SpectralSubtractionAec {
    /// Smoothed delay estimate, in samples back from the end of ref_history.
    delay_estimate: f32,
    /// EMA weight for new measurements (`1.0` = no smoothing).
    delay_alpha: f32,
    /// Search window for the cross-correlation, in samples.
    search_min: usize,
    search_max: usize,
    /// Coarse-search stride (samples). The fine pass refines ±stride at
    /// single-sample resolution around the coarse argmax.
    coarse_stride: usize,
    /// Below this RMS the reference is treated as silent — pass-through.
    ref_silence_rms: f32,
    /// Clip the LSQ gain to this absolute value. Echo coupling is
    /// physically <1.0; large gains usually mean spurious correlation
    /// with user speech that happens to look like the reference.
    gain_clip: f32,
    /// Internal buffer of recent mic samples. Cross-correlation search
    /// and LSQ gain estimate run over this whole buffer, not just the
    /// current frame. Sized at `search_window` samples.
    mic_history: VecDeque<f32>,
    /// Length of the search/estimation window in samples.
    search_window: usize,
    /// Minimum cosine similarity (mic, ref_aligned) for a delay
    /// measurement to update the EMA. Filters spurious peaks during
    /// silence and noisy stretches.
    confidence_threshold: f32,
    /// Once a confident lock is established, the search is restricted
    /// to ±lock_radius around the current estimate. Stops the search
    /// from wandering to wildly wrong delays during low-SNR transients
    /// (e.g. a speech onset where the 2048-sample mic buffer still
    /// holds mostly silence).
    lock_radius: usize,
    /// True after the first confident measurement; gates the narrow
    /// search behavior.
    locked: bool,
    last_stats: Option<FrameStats>,
}

impl Default for SpectralSubtractionAec {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectralSubtractionAec {
    pub fn new() -> Self {
        Self {
            delay_estimate: 0.0,
            // Slow EMA: with the long search window per-frame estimates
            // are already much more reliable, so we lean on smoothing to
            // ride out occasional spurious frames (speech onsets, etc.).
            delay_alpha: 0.05,
            search_min: 0,
            // 250 ms covers playback buffer + room propagation in any
            // reasonable speaker/mic geometry.
            search_max: (SR as f32 * 0.25) as usize,
            coarse_stride: 4,
            ref_silence_rms: 1e-3,
            gain_clip: 2.0,
            mic_history: VecDeque::new(),
            // 128 ms is long enough to wash out pitch-period
            // autocorrelation peaks (pitch ≈ 80–300 Hz, period ≤ 12 ms).
            search_window: 2048,
            // 0.3 keeps confident peaks (echo lock-in is usually >0.5
            // even with noise) while rejecting spurious matches in
            // silent/non-echo regions.
            confidence_threshold: 0.3,
            // ±100 samples (~6 ms) around the current estimate is wide
            // enough for any sane physical delay drift but tight enough
            // that a transient frame can't pick a wildly wrong peak.
            lock_radius: 100,
            locked: false,
            last_stats: None,
        }
    }

    /// Configure the cross-correlation search window. `min..=max` are
    /// in samples. Useful for tests that want a narrow, deterministic
    /// search.
    pub fn with_search_range(mut self, min: usize, max: usize) -> Self {
        self.search_min = min;
        self.search_max = max;
        self
    }

    /// Current smoothed delay estimate (samples back from end of history).
    /// For diagnostics / logging.
    pub fn delay_estimate(&self) -> f32 {
        self.delay_estimate
    }

    /// Per-frame stats from the most recent `process()` call. For
    /// diagnostics / verbose mode in `aec_check`.
    pub fn last_frame_stats(&self) -> Option<&FrameStats> {
        self.last_stats.as_ref()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    pub best_d: usize,
    pub confidence: f32,
    pub gain: f32,
    pub mic_energy: f32,
    pub ref_energy: f32,
}

impl AecKernel for SpectralSubtractionAec {
    fn process(&mut self, mic_frame: &[f32], ref_history: &[f32]) -> Vec<f32> {
        let n = mic_frame.len();
        if n == 0 {
            return Vec::new();
        }
        let w = self.search_window;

        // Update the internal mic-history buffer.
        for &s in mic_frame {
            if self.mic_history.len() == w {
                self.mic_history.pop_front();
            }
            self.mic_history.push_back(s);
        }

        // Need enough history on both sides to align the long window at
        // every candidate delay.
        if self.mic_history.len() < w || ref_history.len() < w + self.search_max {
            return mic_frame.to_vec();
        }

        let mic_buf: Vec<f32> = self.mic_history.iter().copied().collect();

        // Cheap reference-silence + mic-silence early exit. If either
        // side is essentially silent, neither the delay estimate nor a
        // gain estimate is reliable — pass mic through and freeze the
        // delay. The most-recent ref window (delay = current estimate)
        // is a good proxy for "is ref active right now".
        let est_d = self
            .delay_estimate
            .round()
            .clamp(self.search_min as f32, self.search_max as f32) as usize;
        let probe_end = ref_history.len() - est_d;
        let probe_start = probe_end - w;
        let probe_ref = &ref_history[probe_start..probe_end];
        let probe_ref_energy = dot(probe_ref, probe_ref);
        let probe_ref_rms = (probe_ref_energy / w as f32).sqrt();
        if probe_ref_rms < self.ref_silence_rms {
            return mic_frame.to_vec();
        }

        // Search range: full [search_min, search_max] before lock,
        // narrow ±lock_radius around the current estimate after.
        let (range_lo, range_hi) = if self.locked {
            let center = self.delay_estimate.round() as usize;
            let lo = center.saturating_sub(self.lock_radius).max(self.search_min);
            let hi = (center + self.lock_radius).min(self.search_max);
            (lo, hi)
        } else {
            (self.search_min, self.search_max)
        };

        // Coarse search.
        let mut best_d = range_lo;
        let mut best_abs_corr = f32::NEG_INFINITY;
        let mut d = range_lo;
        while d <= range_hi {
            let end = ref_history.len() - d;
            let start = end - w;
            let corr = dot(&mic_buf, &ref_history[start..end]);
            let abs_corr = corr.abs();
            if abs_corr > best_abs_corr {
                best_abs_corr = abs_corr;
                best_d = d;
            }
            d += self.coarse_stride;
        }

        // Fine search ±coarse_stride at single-sample resolution.
        let lo = best_d.saturating_sub(self.coarse_stride).max(range_lo);
        let hi = (best_d + self.coarse_stride).min(range_hi);
        for d in lo..=hi {
            let end = ref_history.len() - d;
            let start = end - w;
            let corr = dot(&mic_buf, &ref_history[start..end]);
            let abs_corr = corr.abs();
            if abs_corr > best_abs_corr {
                best_abs_corr = abs_corr;
                best_d = d;
            }
        }

        // Confidence = cosine similarity at best_d. By Cauchy–Schwarz,
        // this is in [0, 1]: 1 when the alignment is exact, near 0 when
        // the search is matching noise. Gating the EMA update on this
        // stops the delay from drifting in low-SNR stretches.
        let best_end = ref_history.len() - best_d;
        let best_start = best_end - w;
        let best_ref_energy = dot(
            &ref_history[best_start..best_end],
            &ref_history[best_start..best_end],
        );
        let mic_energy = dot(&mic_buf, &mic_buf);
        let confidence = if mic_energy > 0.0 && best_ref_energy > 0.0 {
            best_abs_corr / (mic_energy * best_ref_energy).sqrt()
        } else {
            0.0
        };
        // Two roles for the delay: which one to use this frame, and the
        // EMA that survives between frames. When the measurement is
        // confident, use best_d directly (exact alignment for this
        // frame's subtraction) AND fold it into the EMA. When it isn't,
        // ride the EMA's previous good value and don't poison it.
        let confident = confidence >= self.confidence_threshold;
        let d_use = if confident {
            if self.delay_estimate == 0.0 {
                self.delay_estimate = best_d as f32;
            } else {
                self.delay_estimate = (1.0 - self.delay_alpha) * self.delay_estimate
                    + self.delay_alpha * best_d as f32;
            }
            self.locked = true;
            best_d
        } else {
            self.delay_estimate
                .round()
                .clamp(self.search_min as f32, self.search_max as f32) as usize
        };

        // LSQ gain over the long window: g = <mic_buf, ref_long> / <ref_long, ref_long>.
        let long_end = ref_history.len() - d_use;
        let long_start = long_end - w;
        let ref_long = &ref_history[long_start..long_end];
        let ref_energy = dot(ref_long, ref_long);
        let cross = dot(&mic_buf, ref_long);
        let gain = (cross / ref_energy.max(1e-9)).clamp(-self.gain_clip, self.gain_clip);

        self.last_stats = Some(FrameStats {
            best_d,
            confidence,
            gain,
            mic_energy,
            ref_energy: probe_ref_energy,
        });

        // Subtract from the current frame using the aligned ref window
        // for that frame (the last `n` samples of the long window).
        let frame_end = ref_history.len() - d_use;
        let frame_start = frame_end - n;
        let ref_aligned_frame = &ref_history[frame_start..frame_end];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(mic_frame[i] - gain * ref_aligned_frame[i]);
        }
        out
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Bounded ring buffer of recent reference samples. Producers push from
/// the UDP listener; the AEC kernel reads a snapshot per mic frame.
pub struct ReferenceHistory {
    samples: VecDeque<f32>,
    capacity: usize,
}

impl ReferenceHistory {
    /// Capacity in samples. ~3 s × 16 kHz = 48000 is plenty for any
    /// realistic echo delay.
    pub fn new(capacity_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity_samples),
            capacity: capacity_samples,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            if self.samples.len() == self.capacity {
                self.samples.pop_front();
            }
            self.samples.push_back(s);
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Oldest-first snapshot of the contents. Cheap (~50 KB / 3 s) and
    /// avoids holding the lock across the AEC kernel's compute.
    pub fn snapshot(&self) -> Vec<f32> {
        self.samples.iter().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cheap deterministic LCG, used so synthesis tests don't drag in a
    /// crate. Seeded white-noise gives a delta-like autocorrelation
    /// (sharp xcorr peak) and is uncorrelated to an independent stream
    /// (clean LSQ gain in the user+echo case).
    fn lcg_noise(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // High 32 bits, reinterpret as signed → zero-mean ~U[-1, 1].
                let v = (s >> 32) as u32 as i32 as f32 / i32::MAX as f32;
                v * amp
            })
            .collect()
    }

    /// Energy after AEC has fully engaged. AEC needs
    /// `search_window + search_max` samples of history before it'll run
    /// (default search_window=2048, search_max=4000); we skip past that
    /// plus a few frames for the delay EMA to converge.
    const WARM_SAMPLES: usize = 7_000;

    #[test]
    fn pure_echo_is_cancelled() {
        // Mic = reference shifted by `delay` and scaled by `gain`; no
        // user speech. AEC should null this out almost completely.
        let n_total = 16_000;
        let reference = lcg_noise(n_total, 0xC0FFEE, 0.4);
        let delay = 800; // 50 ms
        let gain = 0.6;
        let mut mic = vec![0f32; n_total];
        for i in delay..n_total {
            mic[i] = reference[i - delay] * gain;
        }

        let mut history = ReferenceHistory::new(48_000);
        let mut kernel = SpectralSubtractionAec::new();
        let chunk = 320;

        let mut mic_residual = 0f32;
        let mut clean_residual = 0f32;
        let mut pos = 0;
        while pos + chunk <= n_total {
            history.push(&reference[pos..pos + chunk]);
            let snap = history.snapshot();
            let cleaned = kernel.process(&mic[pos..pos + chunk], &snap);
            if pos >= WARM_SAMPLES {
                for i in 0..chunk {
                    mic_residual += mic[pos + i] * mic[pos + i];
                    clean_residual += cleaned[i] * cleaned[i];
                }
            }
            pos += chunk;
        }

        let suppression_db = 10.0 * (mic_residual / clean_residual.max(1e-12)).log10();
        assert!(
            suppression_db > 30.0,
            "expected >30 dB echo suppression, got {suppression_db:.1} dB \
             (mic_residual={mic_residual:.4}, clean_residual={clean_residual:.4})"
        );
    }

    #[test]
    fn user_speech_passes_through_when_reference_is_silent() {
        let n = 4000;
        let mic: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * (i as f32 / SR as f32)).sin() * 0.5)
            .collect();
        let reference = vec![0f32; n];

        let mut history = ReferenceHistory::new(48_000);
        let mut kernel = SpectralSubtractionAec::new();
        let chunk = 320;

        let mut max_diff = 0f32;
        let mut pos = 0;
        while pos + chunk <= n {
            history.push(&reference[pos..pos + chunk]);
            let snap = history.snapshot();
            let cleaned = kernel.process(&mic[pos..pos + chunk], &snap);
            for i in 0..chunk {
                max_diff = max_diff.max((mic[pos + i] - cleaned[i]).abs());
            }
            pos += chunk;
        }
        assert!(
            max_diff < 1e-6,
            "silent reference should pass mic through unchanged, max_diff={max_diff}"
        );
    }

    #[test]
    fn ring_buffer_respects_capacity() {
        let mut hist = ReferenceHistory::new(100);
        let v: Vec<f32> = (0..150).map(|i| i as f32).collect();
        hist.push(&v);
        assert_eq!(hist.len(), 100);
        let snap = hist.snapshot();
        assert_eq!(snap[0], 50.0);
        assert_eq!(snap[99], 149.0);
    }

    #[test]
    fn user_plus_echo_keeps_user() {
        // mic = user + echo(reference). User and reference are
        // independent white noise → uncorrelated LSQ gain estimate;
        // cleaned should be much closer to user than mic was.
        let n_total = 16_000;
        let reference = lcg_noise(n_total, 0xC0FFEE, 0.4);
        let user = lcg_noise(n_total, 0xBADBEEF, 0.3);
        let delay = 800;
        let echo_gain = 0.5;
        let mut mic = user.clone();
        for i in delay..n_total {
            mic[i] += reference[i - delay] * echo_gain;
        }

        let mut history = ReferenceHistory::new(48_000);
        let mut kernel = SpectralSubtractionAec::new();
        let chunk = 320;

        let mut user_to_mic = 0f32;
        let mut user_to_clean = 0f32;

        let mut pos = 0;
        while pos + chunk <= n_total {
            history.push(&reference[pos..pos + chunk]);
            let snap = history.snapshot();
            let cleaned = kernel.process(&mic[pos..pos + chunk], &snap);
            if pos >= WARM_SAMPLES {
                for i in 0..chunk {
                    let dm = mic[pos + i] - user[pos + i];
                    let dc = cleaned[i] - user[pos + i];
                    user_to_mic += dm * dm;
                    user_to_clean += dc * dc;
                }
            }
            pos += chunk;
        }
        assert!(
            user_to_clean < user_to_mic * 0.1,
            "expected cleaned to be ≥10× closer to user signal; \
             user_to_mic={user_to_mic:.4}, user_to_clean={user_to_clean:.4}"
        );
    }
}
