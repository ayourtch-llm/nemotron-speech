//! tag_live — dual-stream speaker-attributed ASR for the Nabu voice loop.
//!
//! Transcribes BOTH RTP streams at once (mic on --mic-listen, the TTS reference
//! on --ref-listen) and tags each mic word USER vs ECHO by matching it against
//! the reference words in TIME (echo lags the reference by ~delay) AND CONTENT
//! (fuzzy). Live color-coded view; USER words forwarded to the agent (--text-out).
//!
//! Each stream runs its own StreamingPipeline on its OWN OS THREAD, so the two
//! ASRs use separate CPU cores (measured ~1.78× vs serial; batching the two into
//! one forward is *slower* on CPU — see bench_batch). The main thread does the
//! lightweight matching.
//!
//! TIMING: each decoded word is stamped with the RTP TIMESTAMP of the audio it
//! came from (frame → pushed-sample → nearest packet's RTP ts), not wall-clock —
//! so the time is immune to processing lag and gap-accurate (the sender's ts
//! reflects silence gaps). Both streams share one origin (the replay derives ts
//! from the recorded t_us; LIVE needs the device+kokoro RTP clocks reconciled —
//! see below). Result on the faithful replay: precision held (0 user false-drops
//! on the counting capture), echo recall ~75% on the AGC-off leak capture (up
//! from ~50% with wall-clock timing). Remaining recall loss is window/ASR-variance
//! tuning. LIVE TODO: the device mic and kokoro reference use independent RTP ts
//! origins — estimate the constant offset (echo delay / first-match) to align.

use std::collections::VecDeque;
use std::net::UdpSocket;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;

use nemotron_speech::features::{IncrementalMelExtractor, MelConfig};
use nemotron_speech::model::encoder::FastConformerEncoder;
use nemotron_speech::model::joint::JointNet;
use nemotron_speech::model::predict::PredictNet;
use nemotron_speech::model::ModelConfig;
use nemotron_speech::streaming::StreamingPipeline;
use nemotron_speech::tokenizer::Tokenizer;

/// Encoder frame = mel hop (160) × subsample (8) = 1280 samples (80 ms @ 16 kHz).
const SAMPLES_PER_FRAME: usize = 1280;
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RST: &str = "\x1b[0m";

#[derive(Parser, Debug)]
#[command(about = "Dual-stream (mic + TTS reference) speaker-attributed ASR with per-word USER/ECHO tagging.")]
struct Args {
    #[arg(long)]
    st: std::path::PathBuf,
    #[arg(long)]
    tok: std::path::PathBuf,
    #[arg(long, default_value = "0.0.0.0:9992")]
    mic_listen: String,
    #[arg(long, default_value = "0.0.0.0:9993")]
    ref_listen: String,
    /// Forward USER-tagged words to this UDP target (the agent). Unset = print only.
    #[arg(long)]
    text_out: Option<String>,
    #[arg(long, default_value_t = false)]
    cpu: bool,
    #[arg(long, default_value_t = 2)]
    chunk_batch: usize,
    #[arg(long, default_value_t = 50)]
    delay_lo_ms: i64,
    #[arg(long, default_value_t = 1100)]
    delay_hi_ms: i64,
    #[arg(long, default_value_t = 0.55)]
    fuzzy_thr: f32,
    /// Max ms a mic word waits for the reference before committing its tag.
    #[arg(long, default_value_t = 1500)]
    hold_ms: u64,
    /// Group USER words and send them to --text-out as one utterance after this
    /// many ms with no new USER word (so the agent gets whole sentences, not
    /// per-word fragments). 0 = send each word immediately.
    #[arg(long, default_value_t = 700)]
    flush_ms: u64,
}

fn pick_device(cpu: bool) -> Device {
    if cpu {
        return Device::Cpu;
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "metal"))]
    {
        Device::Cpu
    }
}

fn build_pipeline(st: &std::path::Path, device: &Device, dtype: DType, batch: usize) -> Result<StreamingPipeline> {
    let mel_cfg = MelConfig::nemotron_default();
    let mel = IncrementalMelExtractor::from_safetensors(st, mel_cfg.clone())?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[st.to_path_buf()], dtype, device).context("safetensors")?
    };
    let cfg = ModelConfig::nemotron_06b();
    let encoder = FastConformerEncoder::new(vb.pp("encoder"), cfg.clone())
        .map_err(|e| anyhow::anyhow!("encoder: {e:#}"))?;
    let predict = PredictNet::new(vb.pp("predict"), &cfg).map_err(|e| anyhow::anyhow!("predict: {e:#}"))?;
    let joint = JointNet::new(vb.pp("joint"), &cfg).map_err(|e| anyhow::anyhow!("joint: {e:#}"))?;
    let mut p = StreamingPipeline::new(encoder, predict, joint, mel, mel_cfg, cfg, device.clone(), dtype)?;
    p.set_max_chunk_batch(batch);
    Ok(p)
}

/// Parse an RTP/L16 datagram → (samples, rtp_timestamp). The RTP timestamp is
/// the sample-clock position of the first sample, set by the sender — a
/// processing-lag-immune time source we thread through to each decoded word.
fn parse_l16(buf: &[u8]) -> Option<(Vec<f32>, u32)> {
    if buf.len() < 14 || (buf[0] >> 6) != 2 {
        return None;
    }
    let ts = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let cc = (buf[0] & 0x0f) as usize;
    let mut hdr = 12 + cc * 4;
    if buf[0] & 0x10 != 0 && buf.len() >= hdr + 4 {
        let words = ((buf[hdr + 2] as usize) << 8) | buf[hdr + 3] as usize;
        hdr += 4 + words * 4;
    }
    if buf.len() <= hdr {
        return None;
    }
    let n = (buf.len() - hdr) / 2;
    let samples = (0..n)
        .map(|i| i16::from_be_bytes([buf[hdr + i * 2], buf[hdr + i * 2 + 1]]) as f32 / 32768.0)
        .collect();
    Some((samples, ts))
}

/// Groups a stream's tokens into words with start timestamps (in-progress word
/// held until the next word starts).
struct WordGrouper {
    tok: Tokenizer,
    buf_tokens: Vec<u32>,
    buf_frames: Vec<usize>,
}

impl WordGrouper {
    fn new(tok: Tokenizer) -> Self {
        Self { tok, buf_tokens: Vec::new(), buf_frames: Vec::new() }
    }

    /// Returns completed (word, start_frame_index). The caller maps the frame
    /// index to an RTP timestamp.
    fn push(&mut self, new_tokens: &[u32], new_frames: &[usize]) -> Vec<(String, usize)> {
        self.buf_tokens.extend_from_slice(new_tokens);
        self.buf_frames.extend_from_slice(new_frames);
        let mut words = Vec::new();
        let mut prev = String::new();
        let mut cur = String::new();
        let mut cur_start = 0usize;
        let mut completed_upto = 0usize;
        for i in 0..self.buf_tokens.len() {
            let full = self.tok.detokenize(&self.buf_tokens[0..=i]).unwrap_or_default();
            let added = if full.len() >= prev.len() { full[prev.len()..].to_string() } else { full.clone() };
            let starts = added.starts_with(' ') || (cur.trim().is_empty() && !added.trim().is_empty());
            if starts && !cur.trim().is_empty() {
                words.push((cur.trim().to_string(), cur_start));
                completed_upto = i;
                cur.clear();
            }
            if cur.trim().is_empty() {
                cur_start = self.buf_frames[i];
            }
            cur.push_str(&added);
            prev = full;
        }
        if completed_upto > 0 {
            self.buf_tokens.drain(0..completed_upto);
            self.buf_frames.drain(0..completed_upto);
        }
        words
    }

    /// Emit the trailing in-progress word (which `push` holds until the next word
    /// starts) once the pipeline has decoded `gap_frames` past it with no new
    /// word — i.e. end of utterance. Without this the last word of every phrase
    /// hangs until the next phrase arrives.
    fn flush_stale(&mut self, decoded_frame: usize, gap_frames: usize) -> Option<(String, usize)> {
        let last = *self.buf_frames.last()?;
        if decoded_frame <= last + gap_frames {
            return None;
        }
        let w = self.tok.detokenize(&self.buf_tokens).unwrap_or_default().trim().to_string();
        let start = self.buf_frames[0];
        self.buf_tokens.clear();
        self.buf_frames.clear();
        if w.is_empty() {
            None
        } else {
            Some((w, start))
        }
    }
}

fn norm(w: &str) -> String {
    w.chars().filter(|c| c.is_ascii_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

fn fuzzy(a: &str, b: &str) -> f32 {
    let (a, b) = (norm(a), norm(b));
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }
    let (sh, lo) = if a.len() <= b.len() { (&a, &b) } else { (&b, &a) };
    if sh.len() >= 2 && (lo.starts_with(sh.as_str()) || lo.ends_with(sh.as_str())) {
        return 0.9;
    }
    if lo.contains(sh.as_str()) {
        return sh.len() as f32 / lo.len() as f32;
    }
    let n = a.len().min(b.len());
    let shared = (0..n).filter(|&i| a.as_bytes()[i] == b.as_bytes()[i]).count();
    shared as f32 / a.len().max(b.len()) as f32
}

/// Send the grouped USER utterance to the agent and clear the buffer.
fn flush_user(buf: &mut Vec<String>, sock: &Option<UdpSocket>, addr: &Option<String>) {
    if buf.is_empty() {
        return;
    }
    if let (Some(s), Some(a)) = (sock, addr) {
        let utt = buf.join(" ");
        let _ = s.send_to(utt.as_bytes(), a);
        eprintln!("{GREEN}{BOLD}-> agent:{RST} {utt}");
    }
    buf.clear();
}

/// Messages from the stream workers to the matcher. Words carry their abs time
/// (RTP-timestamp based). Progress = "this pipeline has DECODED up to abs T",
/// sent every step so the matcher knows the reference's true progress even
/// across silence (decoupled from word emission, which lags + bursts).
enum Msg {
    Ref(String, f32),
    Mic(String, f32),
    RefProgress(f32),
    MicProgress(f32),
}

/// One stream worker: blocking UDP recv → pipeline (silence-padded) → words → tx.
fn run_stream(
    is_mic: bool,
    listen: String,
    st: std::path::PathBuf,
    tok: std::path::PathBuf,
    device: Device,
    batch: usize,
    prog_start: Instant,
    tx: mpsc::Sender<Msg>,
) -> Result<()> {
    let _ = prog_start;
    let sock = UdpSocket::bind(&listen).with_context(|| format!("binding {listen}"))?;
    // DEDICATED RECV THREAD: just recv + parse, never blocked by the ASR. Pushes
    // into an unbounded channel so no UDP packets are dropped while the pipeline
    // grinds (the cause of tag_live missing words the agent sees), and it also
    // buffers cleanly through model load.
    let (atx, arx) = mpsc::channel::<(Vec<f32>, u32)>();
    std::thread::spawn(move || {
        let mut buf = vec![0u8; 65_536];
        loop {
            match sock.recv(&mut buf) {
                Ok(n) => {
                    if let Some(p) = parse_l16(&buf[..n]) {
                        if atx.send(p).is_err() {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });
    let mut pipe = build_pipeline(&st, &device, DType::F32, batch)?;
    let mut grouper = WordGrouper::new(Tokenizer::from_file(&tok)?);
    // Anchors map this pipeline's cumulative pushed-sample position to the RTP
    // timestamp of the audio there — so a decoded word's frame → pushed-sample →
    // RTP timestamp (in ms). Gap-accurate and immune to processing lag.
    let mut anchors: VecDeque<(usize, u32)> = VecDeque::new();
    let mut pushed: usize = 0;
    while let Ok((samples, ts)) = arx.recv() {
        if samples.is_empty() {
            continue;
        }
        anchors.push_back((pushed, ts));
        if anchors.len() > 4096 {
            anchors.pop_front();
        }
        pipe.push_audio(&samples);
        pushed += samples.len();
        // frame → abs ms via the nearest anchor at-or-before its pushed-sample.
        let frame_to_abs = |frame: usize| -> f32 {
            let sample = frame * SAMPLES_PER_FRAME;
            let ts_at = anchors
                .iter()
                .rev()
                .find(|(p, _)| *p <= sample)
                .map(|(p, t)| t.wrapping_add((sample - p) as u32))
                .unwrap_or(ts);
            ts_at as f32 / 16.0 // 16 samples / ms @ 16 kHz
        };
        let mut advanced = false;
        while let Some(toks) = pipe.try_advance()? {
            advanced = true;
            if toks.is_empty() {
                continue;
            }
            let prev = pipe.all_tokens.len() - toks.len();
            let frames = pipe.all_frames[prev..].to_vec();
            for (w, frame) in grouper.push(&toks, &frames) {
                let abs_ms = frame_to_abs(frame);
                let msg = if is_mic { Msg::Mic(w, abs_ms) } else { Msg::Ref(w, abs_ms) };
                if tx.send(msg).is_err() {
                    return Ok(());
                }
            }
        }
        if advanced {
            let decoded = pipe.decoded_frames();
            // Flush a trailing word once decoding has moved ~0.5s past it.
            if let Some((w, frame)) = grouper.flush_stale(decoded, 6) {
                let abs_ms = frame_to_abs(frame);
                let msg = if is_mic { Msg::Mic(w, abs_ms) } else { Msg::Ref(w, abs_ms) };
                if tx.send(msg).is_err() {
                    return Ok(());
                }
            }
            // Heartbeat: how far this pipeline has decoded (on the same RTP-
            // timestamp axis as the words), so the matcher knows progress without
            // waiting on emitted words — drives both echo-match readiness (ref)
            // and the utterance-flush idle gap (mic).
            let prog = frame_to_abs(decoded);
            let _ = tx.send(if is_mic { Msg::MicProgress(prog) } else { Msg::RefProgress(prog) });
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = pick_device(args.cpu);
    eprintln!("tag_live: device {:?}", device);

    let prog_start = Instant::now();
    let (tx, rx) = mpsc::channel::<Msg>();

    // One OS thread per stream → true parallelism across cores.
    {
        let (st, tok, dev, tx) = (args.st.clone(), args.tok.clone(), device.clone(), tx.clone());
        let listen = args.ref_listen.clone();
        let b = args.chunk_batch;
        std::thread::spawn(move || {
            if let Err(e) = run_stream(false, listen, st, tok, dev, b, prog_start, tx) {
                eprintln!("ref worker died: {e:#}");
            }
        });
    }
    {
        let (st, tok, dev) = (args.st.clone(), args.tok.clone(), device.clone());
        let listen = args.mic_listen.clone();
        let b = args.chunk_batch;
        std::thread::spawn(move || {
            if let Err(e) = run_stream(true, listen, st, tok, dev, b, prog_start, tx) {
                eprintln!("mic worker died: {e:#}");
            }
        });
    }

    let out_sock = match &args.text_out {
        Some(_) => Some(UdpSocket::bind("0.0.0.0:0")?),
        None => None,
    };
    eprintln!("tag_live: mic {} | ref {} | {GREEN}USER{RST}/{RED}ECHO{RST}\n", args.mic_listen, args.ref_listen);

    let mut ref_recent: VecDeque<(String, f32)> = VecDeque::new();
    let mut pending_mic: VecDeque<(String, f32, Instant)> = VecDeque::new();
    // How far the reference pipeline has DECODED (heartbeat), and when that last
    // advanced — drives both "is the reference caught up to this mic word" and
    // "is TTS active" without depending on word emission timing.
    let mut ref_processed_ms: f32 = 0.0;
    let mut last_ref_progress = Instant::now();
    let hold = Duration::from_millis(args.hold_ms);
    // USER-word grouping for the agent feed: accumulate, flush as one utterance
    // on an AUDIO-time gap (the mic decoded flush_ms past the last USER word, or
    // a new word jumps >flush_ms ahead) so the agent gets whole sentences.
    // Audio-time (RTP ts) not wall-clock, because the pipeline emits in bursts.
    let mut user_buf: Vec<String> = Vec::new();
    let mut last_user_abs: f32 = 0.0;
    let mut mic_processed_ms: f32 = 0.0;
    let flush_gap = args.flush_ms as f32;

    loop {
        match rx.recv_timeout(Duration::from_millis(40)) {
            Ok(Msg::Ref(w, abs)) => {
                println!("{DIM}{RED}[ref ]{RST}{RED} {:>7.1}s  {w}{RST}", abs / 1000.0);
                ref_recent.push_back((w, abs));
            }
            Ok(Msg::RefProgress(abs)) => {
                ref_processed_ms = ref_processed_ms.max(abs);
                last_ref_progress = Instant::now();
            }
            Ok(Msg::MicProgress(abs)) => mic_processed_ms = mic_processed_ms.max(abs),
            Ok(Msg::Mic(w, abs)) => {
                // Stage 1 — raw ASR, printed the instant the word appears.
                println!("{GREEN}[mic ]{RST} {:>7.1}s  {w}", abs / 1000.0);
                pending_mic.push_back((w, abs, Instant::now()));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        let now = Instant::now();
        let tts_active = now.duration_since(last_ref_progress).as_millis() < 600;
        while let Some((w, t, arrived)) = pending_mic.front().cloned() {
            // Commit once the reference pipeline has DECODED past this word's
            // echo-source time (so ref_recent holds the matching word), or TTS is
            // silent, or we time out.
            let ref_caught_up = ref_processed_ms >= t - args.delay_lo_ms as f32;
            let timed_out = now.duration_since(arrived) >= hold;
            if !(ref_caught_up || !tts_active || timed_out) {
                break;
            }
            pending_mic.pop_front();
            let mut best = ("".to_string(), 0.0f32);
            for (rw, rt) in ref_recent.iter() {
                let dt = t - rt;
                if dt >= args.delay_lo_ms as f32 && dt <= args.delay_hi_ms as f32 {
                    let s = fuzzy(&w, rw);
                    if s > best.1 {
                        best = (rw.clone(), s);
                    }
                }
            }
            let is_echo = best.1 >= args.fuzzy_thr;
            let (col, tag) = if is_echo { (RED, "ECHO") } else { (GREEN, "USER") };
            // Stage 2 — the dedup verdict (may lag the raw print above).
            println!(
                "{col}{BOLD}[dup ]{RST}{col} {:>7.1}s  {w:<14} {tag}{RST} {DIM}(ref~'{}' {:.2}){RST}",
                t / 1000.0, best.0, best.1
            );
            // Only forward real words to the agent — skip standalone punctuation
            // tokens (".", "?", …) the ASR emits, which otherwise become junk
            // one-token utterances that trigger spurious turns.
            if !is_echo && args.text_out.is_some() && !norm(&w).is_empty() {
                if args.flush_ms == 0 {
                    if let (Some(sock), Some(addr)) = (&out_sock, &args.text_out) {
                        let _ = sock.send_to(format!("{w} ").as_bytes(), addr);
                    }
                } else {
                    // Word-gap split: a USER word whose audio time jumps >flush_ms
                    // past the last one starts a new utterance — flush first.
                    if !user_buf.is_empty() && t - last_user_abs > flush_gap {
                        flush_user(&mut user_buf, &out_sock, &args.text_out);
                    }
                    user_buf.push(w.clone());
                    last_user_abs = t;
                }
            }
        }

        // Idle flush: the mic has decoded flush_ms past the last USER word with
        // nothing new (the user stopped) — send the grouped utterance.
        if !user_buf.is_empty() && mic_processed_ms - last_user_abs >= flush_gap {
            flush_user(&mut user_buf, &out_sock, &args.text_out);
        }

        let horizon = (args.delay_hi_ms + 4000) as f32;
        while let Some((_, rt)) = ref_recent.front() {
            if ref_processed_ms - rt > horizon {
                ref_recent.pop_front();
            } else {
                break;
            }
        }
    }
    Ok(())
}
