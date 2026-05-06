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
#[cfg(feature = "webrtc-aec")]
use nemotron_speech::aec::WebrtcAec;
use nemotron_speech::aec::{
    AecKernel, FrameStats, NlmsAec, ReferenceHistory, SpectralSubtractionAec,
};
use nemotron_speech::audio::load_audio_mono_16k;
#[cfg(feature = "mic")]
use nemotron_speech::audio_source::mic::MicSource;
use nemotron_speech::audio_source::{AudioSource, FileChunkSource, UdpSource};
use nemotron_speech::features::{IncrementalMelExtractor, MelConfig};
use nemotron_speech::model::ModelConfig;
use nemotron_speech::model::encoder::FastConformerEncoder;
use nemotron_speech::model::joint::JointNet;
use nemotron_speech::model::predict::PredictNet;
use nemotron_speech::streaming::StreamingPipeline;
use nemotron_speech::tokenizer::Tokenizer;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
    /// Mirror each emitted chunk as plain text (newline-terminated) to a UDP
    /// target. Pairs with `nc -lu <port>` for testing or with an LLM-side
    /// consumer that wants to splice user words into its KV cache.
    #[arg(long)]
    text_out: Option<String>,
    /// With `--text-out`, buffer chunks within a single utterance and emit
    /// one UDP datagram per utterance (when the idle-flush fires or
    /// the stream ends). Default: false — each chunk is sent immediately
    /// as it's emitted, which is friendlier for `nc` testing but produces
    /// per-word splices on the LLM side.
    #[arg(long, default_value_t = false)]
    coalesce_text: bool,
    /// Idle threshold (ms) after which a held-back partial-word tail is
    /// flushed and (with --coalesce-text) the buffered utterance is shipped.
    /// Default 1500 ms — comfortable for natural speech pauses (mid-sentence
    /// breaths don't fragment utterances). Lower values give snappier display
    /// at the cost of more fragmentation.
    #[arg(long, default_value_t = 1500)]
    idle_flush_ms: u64,
    /// Bind a UDP socket for the TTS reference signal (raw f32-LE 16 kHz mono
    /// PCM, same wire format as `--udp-listen`). When set, AEC subtracts the
    /// speaker's audio from the mic before transcription. Defaults to off.
    /// See docs/specs/m3-5-echo-cancellation.md.
    #[arg(long)]
    reference_listen: Option<String>,
    /// On startup, POST to this URL to ask speak-server to emit a short
    /// calibration phrase ("Recognition ready.") through both the speaker
    /// and the reference UDP stream. Gives the AEC kernel a clean
    /// delay/gain lock before the noisy bidirectional flow starts.
    /// Best-effort: failures are logged, not fatal. Only meaningful with
    /// `--reference-listen`. Spec §3 phase A.5.
    #[arg(long)]
    calibrate_url: Option<String>,
    /// AEC algorithm. `nlms` is a 4096-tap normalised-LMS adaptive FIR
    /// modelling the room impulse response; `spectral` is the original
    /// single-tap cross-correlation kernel from phase A. NLMS is
    /// default — phase A's single-tap model produced ~0 dB ERLE in
    /// real rooms because multipath echo doesn't correlate strongly at
    /// any single delay. Use `spectral` only as a fallback if NLMS
    /// misbehaves.
    #[cfg_attr(
        feature = "webrtc-aec",
        arg(long, default_value = "nlms", value_parser = ["nlms", "spectral", "webrtc"])
    )]
    #[cfg_attr(
        not(feature = "webrtc-aec"),
        arg(long, default_value = "nlms", value_parser = ["nlms", "spectral"])
    )]
    aec_kernel: String,
    /// AEC3 stream-delay hint in milliseconds — the round-trip
    /// render-to-capture latency. Without this hint AEC3 runs in
    /// blind-delay mode and ERLE collapses (Andrew's first live test
    /// of `--aec-kernel=webrtc` saw 1.5–6.2 dB; the library's own
    /// `delay_median_ms` was returning None for 50+ seconds — the
    /// classic "host didn't tell me the delay" symptom). 200 ms is
    /// a reasonable default for our setup (cross-correlation
    /// measured ~207 ms = speak-server queue + speaker buffer +
    /// acoustic path); sweep this value to find the best for your
    /// room. Only meaningful with `--aec-kernel webrtc`.
    #[cfg(feature = "webrtc-aec")]
    #[arg(long, default_value_t = 200)]
    webrtc_stream_delay_ms: u16,
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
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            Device::Cpu
        }
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
    let predict =
        PredictNet::new(vb.pp("predict"), &cfg).map_err(|e| anyhow::anyhow!("predict: {e:#}"))?;
    let joint = JointNet::new(vb.pp("joint"), &cfg).map_err(|e| anyhow::anyhow!("joint: {e:#}"))?;
    let tok = Tokenizer::from_file(&args.tok)?;

    let mut pipe =
        StreamingPipeline::new(encoder, predict, joint, mel, mel_cfg, cfg, device, dtype)?;

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

    // Optional AEC: bind a second UDP socket for the TTS reference stream
    // and run echo cancellation on each mic chunk before it hits the
    // pipeline. The listener task pushes samples into a shared ring buffer;
    // the main loop snapshots the buffer per chunk and runs the kernel.
    // Without --reference-listen, both sides are None and the loop is
    // bit-identical to before this milestone.
    let ref_history: Option<Arc<Mutex<ReferenceHistory>>> = match &args.reference_listen {
        None => None,
        Some(addr) => {
            // 3 s of history is plenty for any realistic acoustic delay.
            let history = Arc::new(Mutex::new(ReferenceHistory::new(48_000)));
            let socket = tokio::net::UdpSocket::bind(addr)
                .await
                .with_context(|| format!("binding --reference-listen {addr}"))?;
            eprintln!("reference UDP listening on {}", socket.local_addr()?);
            let h = history.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65_536];
                loop {
                    match socket.recv_from(&mut buf).await {
                        Ok((n, _peer)) => {
                            if n % 4 != 0 {
                                tracing::warn!(
                                    "reference UDP datagram of {n} bytes \
                                     (not a multiple of 4 — expected raw f32 LE)"
                                );
                                continue;
                            }
                            let mut samples = Vec::with_capacity(n / 4);
                            for i in 0..n / 4 {
                                let off = i * 4;
                                samples.push(f32::from_le_bytes([
                                    buf[off],
                                    buf[off + 1],
                                    buf[off + 2],
                                    buf[off + 3],
                                ]));
                            }
                            if let Ok(mut h) = h.lock() {
                                h.push(&samples);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("reference UDP recv error: {e}");
                            break;
                        }
                    }
                }
            });
            Some(history)
        }
    };
    let mut aec_kernel: Option<Box<dyn AecKernel>> = if ref_history.is_some() {
        let kernel: Box<dyn AecKernel> = match args.aec_kernel.as_str() {
            "spectral" => Box::new(SpectralSubtractionAec::new()),
            "nlms" => Box::new(NlmsAec::new()),
            #[cfg(feature = "webrtc-aec")]
            "webrtc" => Box::new(WebrtcAec::new(args.webrtc_stream_delay_ms)?),
            #[cfg(not(feature = "webrtc-aec"))]
            "webrtc" => unreachable!("webrtc-aec feature disabled"),
            // clap's value_parser already restricts the set; the
            // exhaustive match is for the type checker.
            other => unreachable!("clap allowed unexpected --aec-kernel {other}"),
        };
        #[cfg(feature = "webrtc-aec")]
        if args.aec_kernel == "webrtc" {
            eprintln!(
                "AEC kernel: webrtc (stream_delay_ms={})",
                args.webrtc_stream_delay_ms
            );
        } else {
            eprintln!("AEC kernel: {}", args.aec_kernel);
        }
        #[cfg(not(feature = "webrtc-aec"))]
        eprintln!("AEC kernel: {}", args.aec_kernel);
        Some(kernel)
    } else {
        None
    };

    // Per-second AEC stats logger. Tracks aggregate ERLE / confidence /
    // gain over each ~16k-sample mic window and emits one tracing::info
    // line per second so the live operator can see actual suppression
    // numbers instead of guessing. Only present when --reference-listen
    // is set, so non-AEC runs stay quiet.
    let mut aec_logger: Option<AecLogger> = ref_history.as_ref().map(|_| AecLogger::new(SR_HZ));

    // Optional startup calibration: ask speak-server to emit a short
    // phrase through the speaker + reference UDP stream so the AEC
    // kernel has a clean signal to lock onto before the user talks.
    // Best-effort — failures here don't bring the daemon down. Only
    // meaningful with --reference-listen; warn-and-skip otherwise.
    let mut cal_probe: Option<CalibrationProbe> = None;
    match (&args.reference_listen, &args.calibrate_url) {
        (Some(_), Some(url)) => match calibrate_post(url) {
            Ok(()) => {
                eprintln!("calibration POST {url} ok");
                // 2 s probe window — covers the "Recognition ready."
                // utterance plus its tail so the gain estimate
                // averages over a real-room acoustic transient.
                cal_probe = Some(CalibrationProbe::new(std::time::Duration::from_millis(
                    2_000,
                )));
            }
            Err(e) => tracing::warn!("calibration POST {url} failed: {e:#}"),
        },
        (None, Some(_)) => {
            tracing::warn!(
                "--calibrate-url has no effect without --reference-listen; skipping POST"
            );
        }
        _ => {}
    }

    // Optional text-mirroring sink: a UDP socket bound to an ephemeral port
    // that forwards each emitted chunk (plain text + '\n') to a target.
    let text_sink: Option<(std::net::UdpSocket, std::net::SocketAddr)> = match &args.text_out {
        None => None,
        Some(spec) => {
            use std::net::ToSocketAddrs;
            let target = spec
                .to_socket_addrs()
                .with_context(|| format!("resolving --text-out target {spec}"))?
                .next()
                .ok_or_else(|| anyhow::anyhow!("no addresses for {spec}"))?;
            let sock = std::net::UdpSocket::bind("0.0.0.0:0")
                .context("binding ephemeral UDP socket for --text-out")?;
            eprintln!("text-out: sending to {target}");
            Some((sock, target))
        }
    };

    eprintln!("listening... (Ctrl-C to stop)");
    std::io::stderr().flush().ok();

    // Index up to which we've already emitted text. We hold back any trailing
    // partial-word run so the next chunk's pieces can complete it without
    // splitting a word across two log lines.
    let mut emitted_idx: usize = 0;
    // If no new token has been produced for this long, the held-back tail
    // must be a complete word — flush it. Configurable via --idle-flush-ms.
    let idle_flush = std::time::Duration::from_millis(args.idle_flush_ms);
    let mut last_token_time = std::time::Instant::now();
    // With --coalesce-text, accumulate chunks within an utterance here and
    // ship the whole buffer on idle-flush / is_final.
    let mut utterance_buf: String = String::new();

    loop {
        match source.next_chunk().await? {
            None => break,
            Some(chunk) => {
                let is_final = chunk.is_final;
                // AEC: if a reference stream is wired up, snapshot the
                // ring buffer (cheap memcpy, ~50 KB) and run the kernel.
                // The snapshot avoids holding the lock across the kernel.
                let cleaned: Option<Vec<f32>> = match (&ref_history, aec_kernel.as_mut()) {
                    (Some(hist), Some(kernel)) => {
                        let snap = match hist.lock() {
                            Ok(h) => h.snapshot(),
                            Err(_) => Vec::new(),
                        };
                        Some(kernel.process(&chunk.samples, &snap))
                    }
                    _ => None,
                };
                let samples_for_pipe: &[f32] = cleaned.as_deref().unwrap_or(&chunk.samples);
                pipe.push_audio(samples_for_pipe);
                if is_final {
                    pipe.finish();
                }

                // AEC diagnostics. Aggregate frame stats; once per ~1 s
                // of mic input, emit a single INFO line summarising the
                // kernel's behaviour in this window.
                if let (Some(logger), Some(kernel), Some(hist)) = (
                    aec_logger.as_mut(),
                    aec_kernel.as_ref(),
                    ref_history.as_ref(),
                ) {
                    let stats = kernel.last_frame_stats().copied();
                    if logger.observe(&chunk.samples, stats.as_ref()) {
                        let (rb_len, rb_cap) = match hist.lock() {
                            Ok(h) => (h.len(), h.capacity()),
                            Err(_) => (0, 0),
                        };
                        logger.flush(kernel.as_ref(), rb_len, rb_cap);
                    }
                }

                // Calibration-window chain-gain probe. Active for the
                // first ~2 s after the calibration POST returns; logs
                // mic_rms / ref_rms / mic-vs-ref dB once. The gain
                // measurement here IS the chain (speaker amp × room
                // attenuation × mic preamp) — if it sits far from 0 dB,
                // AEC3's "is this real echo?" gate may be disabling
                // adaptation in the live test.
                if let Some(probe) = cal_probe.as_mut() {
                    if probe.observe_mic_and_maybe_flush(&chunk.samples, ref_history.as_deref()) {
                        // Window expired and the line was emitted; drop
                        // the probe so we stop checking the deadline
                        // every iteration.
                        cal_probe = None;
                    }
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
                let mut new_chunk_text = String::new();
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
                    new_chunk_text = new_text.to_string();
                }

                if let Some((sock, target)) = &text_sink {
                    if args.coalesce_text {
                        // Accumulate; ship the whole utterance on idle/final.
                        utterance_buf.push_str(&new_chunk_text);
                        let utterance_done =
                            (idle_long_enough || is_final) && !utterance_buf.is_empty();
                        if utterance_done {
                            let mut payload = Vec::with_capacity(utterance_buf.len() + 1);
                            payload.extend_from_slice(utterance_buf.as_bytes());
                            payload.push(b'\n');
                            if let Err(e) = sock.send_to(&payload, target) {
                                tracing::debug!("text-out send: {e}");
                            }
                            utterance_buf.clear();
                        }
                    } else if !new_chunk_text.is_empty() {
                        // Default: send each chunk's new text immediately.
                        let mut payload = Vec::with_capacity(new_chunk_text.len() + 1);
                        payload.extend_from_slice(new_chunk_text.as_bytes());
                        payload.push(b'\n');
                        if let Err(e) = sock.send_to(&payload, target) {
                            tracing::debug!("text-out send: {e}");
                        }
                    }
                }

                if upto > emitted_idx {
                    emitted_idx = upto;
                }
            }
        }
    }
    eprintln!();
    Ok(())
}

/// Mic sample rate the AEC pipeline assumes. Mirrors `aec::SR` but
/// kept local so the logger doesn't need a candle dep.
const SR_HZ: usize = 16_000;

/// Calibration-window probe: started after the `/calibrate` POST
/// returns, accumulates mic samples received during the next
/// `duration_ms`, and at the end snapshots the same number of recent
/// reference samples. The mic RMS / ref RMS ratio is the actual
/// end-to-end chain gain in this room (speaker amp × room
/// attenuation × mic preamp) — known to be a live-mode failure mode
/// for AEC3's adaptation gate (mic ≫ ref or mic ≪ ref pushes the
/// implied room IR outside ‖h‖ ≤ 1, which the gate uses to decide
/// whether there's "real echo" to cancel).
///
/// Reports once at INFO when the window closes; afterwards the main
/// loop drops the probe so we don't pay the deadline-check cost on
/// every chunk for the rest of the run.
struct CalibrationProbe {
    deadline: std::time::Instant,
    sum_mic_e: f64,
    n_mic_samples: usize,
}

impl CalibrationProbe {
    fn new(window: std::time::Duration) -> Self {
        Self {
            deadline: std::time::Instant::now() + window,
            sum_mic_e: 0.0,
            n_mic_samples: 0,
        }
    }

    /// Accumulate the latest mic frame; if the window has now expired,
    /// emit the calibration line (using `ref_history` for the matched
    /// ref window) and return `true` so the caller drops the probe.
    fn observe_mic_and_maybe_flush(
        &mut self,
        mic: &[f32],
        ref_history: Option<&Mutex<ReferenceHistory>>,
    ) -> bool {
        let now = std::time::Instant::now();
        if now < self.deadline {
            for &x in mic {
                self.sum_mic_e += (x as f64) * (x as f64);
            }
            self.n_mic_samples += mic.len();
            return false;
        }
        // Window closed — compute and report.
        let mic_rms = if self.n_mic_samples > 0 {
            (self.sum_mic_e / self.n_mic_samples as f64).sqrt() as f32
        } else {
            0.0
        };
        // Match the ref-side measurement to the mic sample count by
        // taking the most recent N ref samples from the ring. They're
        // the closest in wall-clock to the calibration utterance.
        let (ref_rms, n_ref) = match ref_history {
            None => (0.0, 0),
            Some(mtx) => match mtx.lock() {
                Err(_) => (0.0, 0),
                Ok(h) => {
                    let snap = h.snapshot();
                    let n = self.n_mic_samples.min(snap.len());
                    if n == 0 {
                        (0.0, 0)
                    } else {
                        let window = &snap[snap.len() - n..];
                        let mut acc = 0f64;
                        for &x in window {
                            acc += (x as f64) * (x as f64);
                        }
                        let rms = (acc / n as f64).sqrt() as f32;
                        (rms, n)
                    }
                }
            },
        };
        let mic_dbfs = if mic_rms > 1e-9 {
            20.0 * mic_rms.log10()
        } else {
            f32::NEG_INFINITY
        };
        let ref_dbfs = if ref_rms > 1e-9 {
            20.0 * ref_rms.log10()
        } else {
            f32::NEG_INFINITY
        };
        let chain_db = if mic_rms > 1e-9 && ref_rms > 1e-9 {
            20.0 * (mic_rms / ref_rms).log10()
        } else {
            f32::NAN
        };
        tracing::info!(
            "calibration: mic={:+.1}dBFS ({:.4} rms) ref={:+.1}dBFS ({:.4} rms) chain={:+.1}dB \
             over {} mic samples / {} ref samples",
            mic_dbfs,
            mic_rms,
            ref_dbfs,
            ref_rms,
            chain_db,
            self.n_mic_samples,
            n_ref,
        );
        true
    }
}

/// Aggregator for per-second AEC stats. Sums frame energies over a
/// window of mic samples; once the window crosses one second, emits a
/// single tracing::info line and resets. ERLE is computed from the
/// summed energies (not averaged dB) so silent frames don't bias the
/// reading.
struct AecLogger {
    /// Mic samples accumulated since the last flush. We log when this
    /// crosses `samples_per_log`.
    samples_in_window: usize,
    samples_per_log: usize,
    /// Total mic energy across the window — surfaced as dBFS even when
    /// the kernel didn't engage, so the operator can read mic levels
    /// during ref-silent stretches.
    sum_mic_window_energy: f64,
    /// Aggregated frame energies — only over frames where the kernel
    /// engaged (ref above silence floor). ERLE = 10·log10(mic/cleaned).
    /// `ref_frame_energy` accumulator backs the chain-gain (mic/ref dB)
    /// readout — Andrew's diagnostic for "is AEC3's gate disabling
    /// adaptation because the mic-vs-ref amplitude ratio looks wrong?"
    sum_mic_frame_energy: f32,
    sum_cleaned_frame_energy: f32,
    sum_ref_frame_energy: f32,
    /// Aggregated soft signals — averaged across active frames.
    sum_confidence: f32,
    sum_gain: f32,
    /// Counts.
    active_frames: u32,
    total_frames: u32,
    log_seq: u64,
}

impl AecLogger {
    fn new(sample_rate: usize) -> Self {
        Self {
            samples_in_window: 0,
            samples_per_log: sample_rate,
            sum_mic_window_energy: 0.0,
            sum_mic_frame_energy: 0.0,
            sum_cleaned_frame_energy: 0.0,
            sum_ref_frame_energy: 0.0,
            sum_confidence: 0.0,
            sum_gain: 0.0,
            active_frames: 0,
            total_frames: 0,
            log_seq: 0,
        }
    }

    /// Record a frame's stats. Returns true once the accumulated mic
    /// time crosses `samples_per_log` — the caller should then call
    /// `flush()`. We take the mic samples directly so we can compute
    /// mic dBFS even on frames where the kernel passed through.
    fn observe(&mut self, mic: &[f32], stats: Option<&FrameStats>) -> bool {
        self.samples_in_window += mic.len();
        self.total_frames += 1;
        for &s in mic {
            self.sum_mic_window_energy += (s as f64) * (s as f64);
        }
        if let Some(s) = stats {
            self.sum_mic_frame_energy += s.mic_frame_energy;
            self.sum_cleaned_frame_energy += s.cleaned_frame_energy;
            self.sum_ref_frame_energy += s.ref_frame_energy;
            self.sum_confidence += s.confidence;
            self.sum_gain += s.gain;
            self.active_frames += 1;
        }
        self.samples_in_window >= self.samples_per_log
    }

    fn flush(&mut self, kernel: &dyn AecKernel, ref_buf_len: usize, ref_buf_cap: usize) {
        self.log_seq += 1;
        let delay_ms = kernel.delay_estimate() / 16.0;
        let frames_since_lock = kernel.frames_since_lock();
        let buf_pct = if ref_buf_cap > 0 {
            100.0 * ref_buf_len as f32 / ref_buf_cap as f32
        } else {
            0.0
        };
        // Mic dBFS from the WINDOW's raw samples (always meaningful).
        // Reference 0 dBFS = 1.0 amplitude. -inf if silent.
        let mic_rms = if self.samples_in_window > 0 {
            (self.sum_mic_window_energy / self.samples_in_window as f64).sqrt() as f32
        } else {
            0.0
        };
        let mic_dbfs = if mic_rms > 1e-9 {
            20.0 * mic_rms.log10()
        } else {
            f32::NEG_INFINITY
        };
        if self.active_frames > 0 {
            let erle_db = 10.0
                * (self.sum_mic_frame_energy / self.sum_cleaned_frame_energy.max(1e-12)).log10();
            let avg_conf = self.sum_confidence / self.active_frames as f32;
            let avg_gain = self.sum_gain / self.active_frames as f32;
            // Per-frame mic / ref RMS over the active frames only.
            // These are the values AEC3's gate sees — if the ratio is
            // way off ‖h‖ ≤ 1 (mic louder than ref) it'll disable
            // adaptation regardless of how good the kernel is.
            let active_samples = (self.active_frames as usize)
                * (self.samples_in_window / self.total_frames.max(1) as usize);
            let active_mic_rms = if active_samples > 0 {
                (self.sum_mic_frame_energy / active_samples as f32).sqrt()
            } else {
                0.0
            };
            let active_ref_rms = if active_samples > 0 {
                (self.sum_ref_frame_energy / active_samples as f32).sqrt()
            } else {
                0.0
            };
            let chain_db = if active_ref_rms > 1e-9 && active_mic_rms > 1e-9 {
                20.0 * (active_mic_rms / active_ref_rms).log10()
            } else {
                f32::NAN
            };
            let ref_dbfs = if active_ref_rms > 1e-9 {
                20.0 * active_ref_rms.log10()
            } else {
                f32::NEG_INFINITY
            };
            tracing::info!(
                "aec t={}s ref_buf={}/{} ({:.0}%) active={}/{} delay={:.1}ms gain={:+.3} conf={:.2} ERLE={:+.1}dB mic={:+.1}dBFS ref={:+.1}dBFS chain={:+.1}dB since_lock={}",
                self.log_seq,
                ref_buf_len,
                ref_buf_cap,
                buf_pct,
                self.active_frames,
                self.total_frames,
                delay_ms,
                avg_gain,
                avg_conf,
                erle_db,
                mic_dbfs,
                ref_dbfs,
                chain_db,
                frames_since_lock,
            );
        } else {
            // ref was below the silence floor (or insufficient history)
            // for the whole window — ERLE is meaningless, mic was passed
            // through unchanged. We still print mic dBFS so the operator
            // can confirm the daemon is at least seeing mic input
            // (helpful for diagnosing "ref_buf=0 forever" — is the mic
            // also broken or just the reference path?).
            tracing::info!(
                "aec t={}s ref_buf={}/{} ({:.0}%) ref=silent active=0/{} delay={:.1}ms mic={:+.1}dBFS since_lock={}",
                self.log_seq,
                ref_buf_len,
                ref_buf_cap,
                buf_pct,
                self.total_frames,
                delay_ms,
                mic_dbfs,
                frames_since_lock,
            );
        }
        // Reset window; counters carry forward via log_seq only.
        self.samples_in_window = 0;
        self.sum_mic_window_energy = 0.0;
        self.sum_mic_frame_energy = 0.0;
        self.sum_cleaned_frame_energy = 0.0;
        self.sum_ref_frame_energy = 0.0;
        self.sum_confidence = 0.0;
        self.sum_gain = 0.0;
        self.active_frames = 0;
        self.total_frames = 0;
    }
}

/// One-shot blocking HTTP POST with no body. Hand-rolled because (a) it's
/// literally one request, (b) avoids pulling in an HTTP-client crate for
/// 30 lines of work, and (c) at startup the runtime has nothing else to
/// do, so blocking the executor for the duration is fine. Used by the
/// `--calibrate-url` pathway to ask speak-server for a calibration phrase.
fn calibrate_post(url: &str) -> Result<()> {
    use std::io::{BufRead, BufReader, Write as _};
    use std::net::{TcpStream, ToSocketAddrs};
    use std::time::Duration;

    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| anyhow::anyhow!("--calibrate-url must start with http://"))?;
    let (host_port, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let addr = host_port
        .to_socket_addrs()
        .with_context(|| format!("resolving {host_port}"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("no addresses for {host_port}"))?;

    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
        .with_context(|| format!("connecting to {addr}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;

    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(req.as_bytes()).context("writing POST")?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader
        .read_line(&mut status_line)
        .context("reading status line")?;
    let trimmed = status_line.trim_end();
    let code: u16 = trimmed
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    if !(200..300).contains(&code) {
        anyhow::bail!("HTTP {code}: {trimmed}");
    }
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
