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

#[cfg(feature = "webrtc-aec")]
use webrtc_audio_processing::Config as WebrtcConfig;
#[cfg(feature = "webrtc-aec")]
use webrtc_audio_processing::Processor;
#[cfg(feature = "webrtc-aec")]
use webrtc_audio_processing_config::EchoCanceller;

/// Sample rate everything in this module assumes (matches mic + reference).
pub const SR: u32 = 16_000;

/// Trait for an echo-cancellation algorithm. Implementations are stateful
/// across frames (e.g. they smooth a delay estimate or carry filter taps).
///
/// The diagnostic methods below back the per-second AEC log in
/// `transcribe_live`. Each kernel is free to interpret the fields how it
/// likes — `delay_estimate` is the bulk delay in samples for
/// `SpectralSubtractionAec`, the peak-tap index for `NlmsAec`, etc. The
/// only invariant is that `last_frame_stats` is `None` when the kernel
/// passed mic through unchanged (silent reference, insufficient history,
/// or kernel-specific freeze).
pub trait AecKernel: Send {
    /// Process one mic frame against the most recent reference samples.
    /// `ref_history` is oldest-first; the kernel handles its own
    /// alignment / adaptation. Returns cleaned samples of the same
    /// length as `mic_frame`. On silent reference or insufficient
    /// history, must return `mic_frame.to_vec()` and set
    /// `last_frame_stats` to `None`.
    fn process(&mut self, mic_frame: &[f32], ref_history: &[f32]) -> Vec<f32>;

    /// Stats from the most recent `process()` call, or `None` if the
    /// last call was a pass-through (no useful AEC work to report).
    fn last_frame_stats(&self) -> Option<&FrameStats> {
        None
    }

    /// A kernel-specific "where is the echo" measure. For
    /// `SpectralSubtractionAec` this is the smoothed bulk delay in
    /// samples; for `NlmsAec` it's the peak filter-tap index.
    fn delay_estimate(&self) -> f32 {
        0.0
    }

    /// Frames since the kernel last did a meaningful state update —
    /// confident lock-in for `SpectralSubtractionAec`, last
    /// non-frozen adaptation step for `NlmsAec`.
    fn frames_since_lock(&self) -> u64 {
        0
    }
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
    /// Frames processed since the last confident lock-in update.
    /// Increments every `process()` call; resets to 0 when a confident
    /// measurement updates the EMA. Useful for diagnostics: a growing
    /// count means the kernel is coasting on its last good lock.
    frames_since_lock: u64,
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
            frames_since_lock: 0,
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
}

#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    pub best_d: usize,
    pub confidence: f32,
    pub gain: f32,
    /// Energy of the long mic-history window (search_window samples).
    pub mic_energy: f32,
    /// Energy of the long reference window at the current delay estimate.
    pub ref_energy: f32,
    /// Energy of *just this frame's* mic input — needed for ERLE.
    pub mic_frame_energy: f32,
    /// Energy of *just this frame's* cleaned output — needed for ERLE.
    pub cleaned_frame_energy: f32,
    /// Energy of the most-recent `mic_frame.len()` ref samples — i.e.
    /// the ref samples sharing the same time window as the mic frame.
    /// Used (alongside `mic_frame_energy`) for the chain-gain
    /// diagnostic: AEC3 expects mic ≈ h·ref with ‖h‖ ≤ 1, and a wildly
    /// off mic/ref RMS ratio in our UDP-decoupled chain is a likely
    /// reason live ERLE collapses. Surfacing both makes the ratio
    /// readable in the per-second log.
    pub ref_frame_energy: f32,
}

impl AecKernel for SpectralSubtractionAec {
    fn process(&mut self, mic_frame: &[f32], ref_history: &[f32]) -> Vec<f32> {
        let n = mic_frame.len();
        if n == 0 {
            return Vec::new();
        }
        let w = self.search_window;

        // Diagnostics counter — increments every call, resets only on a
        // confident lock-in below.
        self.frames_since_lock = self.frames_since_lock.saturating_add(1);

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
            // Pass-through — clear any stale stats from prior frames so
            // callers don't read forward-leaked numbers.
            self.last_stats = None;
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
            self.last_stats = None;
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
            self.frames_since_lock = 0;
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

        // Subtract from the current frame using the aligned ref window
        // for that frame (the last `n` samples of the long window).
        let frame_end = ref_history.len() - d_use;
        let frame_start = frame_end - n;
        let ref_aligned_frame = &ref_history[frame_start..frame_end];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(mic_frame[i] - gain * ref_aligned_frame[i]);
        }

        // Frame-level energies for ERLE. Aggregating these (sum-of-energy,
        // not avg-of-dB) over a window gives a meaningful suppression
        // number even when individual frames are noisy.
        let mic_frame_energy = dot(mic_frame, mic_frame);
        let cleaned_frame_energy = dot(&out, &out);
        // Energy of the most-recent `n` ref samples, regardless of the
        // kernel's chosen alignment delay — for chain-gain diagnostic.
        let recent_ref = &ref_history[ref_history.len() - n..];
        let ref_frame_energy = dot(recent_ref, recent_ref);
        self.last_stats = Some(FrameStats {
            best_d,
            confidence,
            gain,
            mic_energy,
            ref_energy: probe_ref_energy,
            mic_frame_energy,
            cleaned_frame_energy,
            ref_frame_energy,
        });

        out
    }

    fn last_frame_stats(&self) -> Option<&FrameStats> {
        self.last_stats.as_ref()
    }

    fn delay_estimate(&self) -> f32 {
        self.delay_estimate
    }

    fn frames_since_lock(&self) -> u64 {
        self.frames_since_lock
    }
}

/// Phase B AEC: normalised LMS adaptive FIR. Models the
/// speaker→air→mic impulse response as a vector of `n_taps` filter
/// coefficients `w[k]` and adapts per sample to minimise the residual
/// `e[n] = mic[n] − sum_k w[k] · ref[n−k]`. Replaces
/// `SpectralSubtractionAec` for real-room use: the single-tap LSQ
/// model can't follow multipath echo (Andrew's M3.5 live test showed
/// 0 dB ERLE in his actual room, with cosine similarity stuck at 0.04
/// — no single delay correlates strongly enough), but a multi-tap FIR
/// just sums the contributions of every reflection.
///
/// Sizing: 4096 taps at 16 kHz = 256 ms. Long enough to cover bulk
/// playback latency + room impulse response (Andrew's calibration
/// peaked at 207 ms; typical RT60 is well under 100 ms in non-reverb
/// spaces). Per-sample cost is dominated by three N-length passes
/// (echo prediction, ‖x‖², coefficient update) ≈ 200 MFLOPs/s — fits
/// CPU comfortably.
///
/// Double-talk handling: a Geigel-style heuristic — if the frame's
/// mic RMS exceeds `double_talk_ratio` × ref RMS, freeze the filter
/// for that frame. The sum-of-squares for echo only is bounded by
/// ‖h‖₂ · ref RMS ≤ ref RMS for any physical (passive) room, so a
/// sustained excess clearly signals user speech mixed in. Updating
/// during double-talk is the classic NLMS divergence mode (filter
/// tries to subtract the user too, ends up amplifying noise) — the
/// kind of failure Andrew explicitly called out.
pub struct NlmsAec {
    /// Filter coefficients. Stored newest-first: `w[0]` is applied to
    /// the most recent reference sample, `w[k]` to ref n−k.
    w: Vec<f32>,
    /// Filter length. Default 4096 samples (~256 ms at 16 kHz).
    n_taps: usize,
    /// NLMS step size. 0 < mu < 2 is the textbook stability range;
    /// 0.5 is a moderate default that converges in a few hundred ms
    /// of speech without obvious overshoot.
    mu: f32,
    /// Numerical floor for the ‖x‖² normalisation — keeps the update
    /// from blowing up when the reference momentarily dips to silence
    /// inside an active stretch.
    delta: f32,
    /// Mic-RMS / ref-RMS ratio above which we freeze adaptation for
    /// the current frame. 1.2 = 20% headroom over ref-only level.
    double_talk_ratio: f32,
    /// Below this RMS the reference is treated as silent — pass-through.
    /// Matches `SpectralSubtractionAec`'s threshold so the diagnostic
    /// log "ref=silent" semantics carry over.
    ref_silence_rms: f32,
    /// Frames since the last frame in which we actually updated the
    /// filter (i.e. ref active and not in double-talk). Mirrors
    /// `SpectralSubtractionAec::frames_since_lock` for the diagnostic
    /// line — large values mean the filter is coasting.
    frames_since_update: u64,
    last_stats: Option<FrameStats>,
}

impl Default for NlmsAec {
    fn default() -> Self {
        Self::new()
    }
}

impl NlmsAec {
    pub fn new() -> Self {
        Self::with_taps(4096)
    }

    pub fn with_taps(n_taps: usize) -> Self {
        assert!(n_taps > 0);
        Self {
            w: vec![0.0; n_taps],
            n_taps,
            mu: 0.5,
            delta: 1e-6,
            double_talk_ratio: 1.2,
            ref_silence_rms: 1e-3,
            frames_since_update: 0,
            last_stats: None,
        }
    }

    /// Set the NLMS step size. Useful for tests that want fast
    /// convergence on stationary signals.
    pub fn with_mu(mut self, mu: f32) -> Self {
        self.mu = mu;
        self
    }

    /// Index of the largest |w[k]|. Useful as a "where is the echo"
    /// readout — a converged filter on a clean delayed echo will peak
    /// at exactly the propagation delay.
    fn peak_tap(&self) -> (usize, f32) {
        let mut idx = 0;
        let mut peak = 0.0f32;
        for (i, &v) in self.w.iter().enumerate() {
            if v.abs() > peak {
                peak = v.abs();
                idx = i;
            }
        }
        (idx, peak)
    }
}

impl AecKernel for NlmsAec {
    fn process(&mut self, mic_frame: &[f32], ref_history: &[f32]) -> Vec<f32> {
        let n = mic_frame.len();
        if n == 0 {
            return Vec::new();
        }
        let n_taps = self.n_taps;

        self.frames_since_update = self.frames_since_update.saturating_add(1);

        // Need n_taps samples of ref history per output sample, plus
        // the n samples of the current frame. Smallest index touched
        // is `ref_history.len() - n - n_taps + 1`, which must be ≥ 0.
        if ref_history.len() < n_taps + n {
            self.last_stats = None;
            return mic_frame.to_vec();
        }

        // ref_pos for mic_frame[i] is ref_history.len() - n + i — i.e.
        // the most recent ref sample arrives in time-step with the
        // newest mic sample. The ref-window for output sample i covers
        // [ref_pos - n_taps + 1, ref_pos].
        let ref_first = ref_history.len() - n;

        // Frame-level RMS for double-talk gating + silence detection.
        let frame_mic_e = dot(mic_frame, mic_frame);
        let frame_mic_rms = (frame_mic_e / n as f32).sqrt();

        // Reference window covering the whole frame (union of per-sample
        // windows). Cheap because it's just one pass — n + n_taps - 1
        // samples for n=320, n_taps=4096 ≈ 4400 samples.
        let ref_block_start = ref_first + 1 - n_taps; // = ref_history.len() - n - n_taps + 1
        let ref_block = &ref_history[ref_block_start..ref_history.len()];
        let ref_block_e = dot(ref_block, ref_block);
        let frame_ref_rms = (ref_block_e / ref_block.len() as f32).sqrt();

        if frame_ref_rms < self.ref_silence_rms {
            self.last_stats = None;
            return mic_frame.to_vec();
        }

        let in_double_talk = frame_mic_rms > self.double_talk_ratio * frame_ref_rms;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let ref_pos = ref_first + i;

            // Echo estimate: y_hat = sum_{k=0..n_taps-1} w[k] · ref[ref_pos - k].
            let mut y_hat = 0.0f32;
            for k in 0..n_taps {
                y_hat += self.w[k] * ref_history[ref_pos - k];
            }
            let e = mic_frame[i] - y_hat;
            out.push(e);

            // Adaptation. Sample-aligned ‖x‖² (sliding window, but we
            // recompute for clarity — the per-sample cost is the same
            // order as y_hat itself; if profiling forces the issue we
            // can swap to a running sum).
            if !in_double_talk {
                let mut x_norm2 = 0.0f32;
                for k in 0..n_taps {
                    let r = ref_history[ref_pos - k];
                    x_norm2 += r * r;
                }
                let step = self.mu * e / (x_norm2 + self.delta);
                for k in 0..n_taps {
                    self.w[k] += step * ref_history[ref_pos - k];
                }
            }
        }

        if !in_double_talk {
            self.frames_since_update = 0;
        }

        let frame_clean_e = dot(&out, &out);
        let (peak_idx, peak_val) = self.peak_tap();

        // FrameStats interpretation for NLMS:
        // - best_d   = peak filter tap (where the IR has its maximum)
        // - gain     = peak |w[k]| (filter strength at that tap)
        // - confidence = 0.0 during double-talk, 1.0 otherwise
        //   (NLMS doesn't have the cross-correlation cosine the
        //    spectral kernel uses; this gives the diagnostic log
        //    a similarly-shaped 0..1 indicator of "is the kernel
        //    confidently adapting right now").
        // - mic_energy / ref_energy keep the long-window meaning
        //   (frame-level here, since NLMS doesn't carry a separate
        //   long buffer).
        // Energy of the most-recent n ref samples — for chain-gain diagnostic.
        let recent_ref = &ref_history[ref_history.len() - n..];
        let ref_frame_energy = dot(recent_ref, recent_ref);
        self.last_stats = Some(FrameStats {
            best_d: peak_idx,
            confidence: if in_double_talk { 0.0 } else { 1.0 },
            gain: peak_val,
            mic_energy: frame_mic_e,
            ref_energy: ref_block_e,
            mic_frame_energy: frame_mic_e,
            cleaned_frame_energy: frame_clean_e,
            ref_frame_energy,
        });

        out
    }

    fn last_frame_stats(&self) -> Option<&FrameStats> {
        self.last_stats.as_ref()
    }

    fn delay_estimate(&self) -> f32 {
        self.peak_tap().0 as f32
    }

    fn frames_since_lock(&self) -> u64 {
        self.frames_since_update
    }
}

/// AEC3 baseline through `webrtc-audio-processing`.
///
/// This is intentionally feature-gated and not the default kernel. It
/// exists as a measurement baseline for the room before the pure-Rust
/// residual suppressor lands on top of NLMS.
///
/// Stream-delay note: the first version of this kernel left the
/// `EchoCanceller::Full { stream_delay_ms }` field at `None`, which
/// puts AEC3 into blind-delay mode. The live test in Andrew's room
/// hit only 1.5–6.2 dB ERLE that way (the library's own
/// `echo_return_loss_enhancement` reading 0.176 dB and
/// `delay_median_ms` returning `None` for 50+ seconds — both signs
/// of the host failing to provide the round-trip render-to-capture
/// hint). Setting this field unblocks the proper AEC3 fast path.
/// Default 200 ms matches the cross-correlation-measured ~207 ms in
/// our setup (speak-server queue + speaker buffer + acoustic path);
/// the CLI exposes a sweep knob in transcribe_live / aec_check.
#[cfg(feature = "webrtc-aec")]
pub struct WebrtcAec {
    processor: Processor,
    stream_delay_ms: u16,
    last_stats: Option<FrameStats>,
    frames_since_lock: u64,
}

#[cfg(feature = "webrtc-aec")]
impl WebrtcAec {
    /// Construct an AEC3 processor with the given render-to-capture
    /// round-trip hint in milliseconds. The crate forwards this to the
    /// FFI `set_stream_delay_ms` per capture frame; we do not need to
    /// touch the processor between frames ourselves.
    pub fn new(stream_delay_ms: u16) -> anyhow::Result<Self> {
        let processor = Processor::new(SR).map_err(|err| anyhow::anyhow!("{err:?}"))?;
        processor.set_config(WebrtcConfig {
            echo_canceller: Some(EchoCanceller::Full {
                stream_delay_ms: Some(stream_delay_ms),
            }),
            high_pass_filter: Some(Default::default()),
            ..Default::default()
        });
        Ok(Self {
            processor,
            stream_delay_ms,
            last_stats: None,
            frames_since_lock: 0,
        })
    }

    /// The stream-delay hint this processor was constructed with.
    /// Used for diagnostic logging at startup.
    pub fn stream_delay_ms(&self) -> u16 {
        self.stream_delay_ms
    }

    fn sample_rate_to_samples(ms: Option<u32>) -> usize {
        ms.unwrap_or(0).max(0) as usize * SR as usize / 1_000
    }
}

#[cfg(feature = "webrtc-aec")]
impl AecKernel for WebrtcAec {
    fn process(&mut self, mic_frame: &[f32], ref_history: &[f32]) -> Vec<f32> {
        let frame_len = self.processor.num_samples_per_frame();
        if mic_frame.is_empty() || frame_len == 0 || mic_frame.len() < frame_len {
            self.last_stats = None;
            return mic_frame.to_vec();
        }

        let n_blocks = mic_frame.len() / frame_len;
        if n_blocks == 0 {
            self.last_stats = None;
            return mic_frame.to_vec();
        }
        let required_ref = n_blocks * frame_len;
        if ref_history.len() < required_ref {
            self.last_stats = None;
            self.frames_since_lock = self.frames_since_lock.saturating_add(1);
            return mic_frame.to_vec();
        }

        let ref_tail = &ref_history[ref_history.len() - required_ref..];
        let mut out = Vec::with_capacity(n_blocks * frame_len);

        for block_idx in 0..n_blocks {
            let start = block_idx * frame_len;
            let end = start + frame_len;

            let mut render_frame = vec![ref_tail[start..end].to_vec()];
            if self
                .processor
                .process_render_frame(&mut render_frame)
                .is_err()
            {
                self.last_stats = None;
                return mic_frame.to_vec();
            }

            let mut capture_frame = vec![mic_frame[start..end].to_vec()];
            if self
                .processor
                .process_capture_frame(&mut capture_frame)
                .is_err()
            {
                self.last_stats = None;
                return mic_frame.to_vec();
            }

            let processed = capture_frame.pop().unwrap_or_default();
            out.extend_from_slice(&processed);
        }

        // Frame-level totals (across all blocks of this mic frame).
        // Stats from the library reflect AEC3's most recent internal
        // update — we keep just the last per-frame snapshot.
        let mic_frame_energy = dot(mic_frame, mic_frame);
        let cleaned_frame_energy = dot(&out, &out);
        let ref_frame_energy = dot(ref_tail, ref_tail);
        let stats = self.processor.get_stats();
        self.last_stats = Some(FrameStats {
            best_d: Self::sample_rate_to_samples(stats.delay_median_ms),
            confidence: stats
                .voice_detected
                .map(|v| if v { 1.0 } else { 0.0 })
                .unwrap_or(0.0),
            gain: stats.echo_return_loss_enhancement.unwrap_or(0.0) as f32,
            mic_energy: mic_frame_energy,
            ref_energy: ref_frame_energy,
            mic_frame_energy,
            cleaned_frame_energy,
            ref_frame_energy,
        });
        self.frames_since_lock = if stats.delay_median_ms.is_some() {
            0
        } else {
            self.frames_since_lock.saturating_add(1)
        };
        out
    }

    fn last_frame_stats(&self) -> Option<&FrameStats> {
        self.last_stats.as_ref()
    }

    fn delay_estimate(&self) -> f32 {
        self.last_stats
            .as_ref()
            .map(|stats| stats.best_d as f32)
            .unwrap_or(0.0)
    }

    fn frames_since_lock(&self) -> u64 {
        self.frames_since_lock
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

    /// NLMS settle period. Filter has 4096 default taps; for a clean
    /// delayed-echo target with white noise input, NLMS converges in
    /// roughly 5–10× N samples = ~30k samples = ~2 s. We push 4 s of
    /// audio and measure ERLE in the last 1 s, well past the
    /// transient.
    const NLMS_TOTAL: usize = 4 * 16_000;
    const NLMS_MEASURE_FROM: usize = 3 * 16_000;

    #[test]
    fn nlms_pure_echo_25db() {
        // Same scenario as `pure_echo_is_cancelled` but with NlmsAec —
        // spec acceptance is ≥25 dB ERLE on this synthetic case.
        let reference = lcg_noise(NLMS_TOTAL, 0xC0FFEE, 0.4);
        let delay = 800;
        let gain = 0.6;
        let mut mic = vec![0f32; NLMS_TOTAL];
        for i in delay..NLMS_TOTAL {
            mic[i] = reference[i - delay] * gain;
        }

        let mut history = ReferenceHistory::new(48_000);
        let mut kernel = NlmsAec::new();
        let chunk = 320;

        let mut mic_residual = 0f32;
        let mut clean_residual = 0f32;
        let mut pos = 0;
        while pos + chunk <= NLMS_TOTAL {
            history.push(&reference[pos..pos + chunk]);
            let snap = history.snapshot();
            let cleaned = kernel.process(&mic[pos..pos + chunk], &snap);
            if pos >= NLMS_MEASURE_FROM {
                for i in 0..chunk {
                    mic_residual += mic[pos + i] * mic[pos + i];
                    clean_residual += cleaned[i] * cleaned[i];
                }
            }
            pos += chunk;
        }

        let suppression_db = 10.0 * (mic_residual / clean_residual.max(1e-12)).log10();
        assert!(
            suppression_db >= 25.0,
            "expected NLMS ≥25 dB ERLE on pure echo, got {suppression_db:.1} dB \
             (mic_residual={mic_residual:.4}, clean_residual={clean_residual:.4})"
        );

        // Sanity: peak filter tap should land at the true delay (±a few
        // samples for stochastic gradient noise).
        let (peak_idx, peak_val) = kernel.peak_tap();
        assert!(
            (peak_idx as i64 - delay as i64).abs() <= 4,
            "expected peak filter tap near delay {delay}, got {peak_idx} (val={peak_val:.3})"
        );
        assert!(
            (peak_val - gain).abs() < 0.1,
            "expected peak filter value near echo gain {gain}, got {peak_val:.3}"
        );
    }

    #[test]
    fn nlms_silent_reference_passes_through() {
        // Mirror of `user_speech_passes_through_when_reference_is_silent`
        // for the NLMS kernel — silent ref must yield mic unchanged.
        let n = 4000;
        let mic: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * (i as f32 / SR as f32)).sin() * 0.5)
            .collect();
        let reference = vec![0f32; n];

        let mut history = ReferenceHistory::new(48_000);
        let mut kernel = NlmsAec::new();
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
    fn nlms_user_plus_echo_keeps_user() {
        // Realistic double-talk pattern: ref (TTS) active throughout;
        // user (near-end speech) intermittent, only present in the
        // back half. The first half lets NLMS converge on the echo
        // IR cleanly; in the second half double-talk freeze should
        // hold the filter steady while the now-converged
        // (mic − filter·ref) leaves the user signal untouched.
        let reference = lcg_noise(NLMS_TOTAL, 0xC0FFEE, 0.4);
        let user_full = lcg_noise(NLMS_TOTAL, 0xBADBEEF, 0.5);
        let user_start = NLMS_TOTAL / 2;
        let mut user = vec![0f32; NLMS_TOTAL];
        user[user_start..].copy_from_slice(&user_full[user_start..]);

        let delay = 800;
        let echo_gain = 0.5;
        let mut mic = user.clone();
        for i in delay..NLMS_TOTAL {
            mic[i] += reference[i - delay] * echo_gain;
        }

        let mut history = ReferenceHistory::new(48_000);
        let mut kernel = NlmsAec::new();
        let chunk = 320;

        let mut user_to_mic = 0f32;
        let mut user_to_clean = 0f32;
        let mut pos = 0;
        while pos + chunk <= NLMS_TOTAL {
            history.push(&reference[pos..pos + chunk]);
            let snap = history.snapshot();
            let cleaned = kernel.process(&mic[pos..pos + chunk], &snap);
            // Measure during the user-active back half, well past the
            // mid-stream double-talk onset.
            if pos >= NLMS_MEASURE_FROM {
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
            "expected NLMS cleaned to be ≥10× closer to user signal during \
             double-talk; user_to_mic={user_to_mic:.4}, \
             user_to_clean={user_to_clean:.4}"
        );
    }
}
