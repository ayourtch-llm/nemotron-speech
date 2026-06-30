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
//! KNOWN LIMITATION (timing): a word's time = stream-start offset + pipeline
//! frame-time — accurate for the continuous mic but it DRIFTS for the reference
//! across TTS silence gaps, and the streaming chunked path can fall behind
//! real-time, so per-word USER/ECHO tagging is currently partial (~50% echo
//! recall on the faithful replay). The mechanism is correct (aligned pairs tag
//! perfectly); the open work is a processing-lag-immune audio clock (RTP
//! timestamp based) + cross-stream sync. See handoff "tag_live timing".

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

const FRAME_MS: f32 = 80.0;
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

/// Parse an RTP/L16 datagram (12-byte header, 16-bit BE PCM) to f32 samples.
fn parse_l16(buf: &[u8]) -> Vec<f32> {
    if buf.len() < 14 || (buf[0] >> 6) != 2 {
        return Vec::new();
    }
    let cc = (buf[0] & 0x0f) as usize;
    let mut hdr = 12 + cc * 4;
    if buf[0] & 0x10 != 0 && buf.len() >= hdr + 4 {
        let words = ((buf[hdr + 2] as usize) << 8) | buf[hdr + 3] as usize;
        hdr += 4 + words * 4;
    }
    if buf.len() <= hdr {
        return Vec::new();
    }
    let n = (buf.len() - hdr) / 2;
    (0..n)
        .map(|i| i16::from_be_bytes([buf[hdr + i * 2], buf[hdr + i * 2 + 1]]) as f32 / 32768.0)
        .collect()
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

    fn push(&mut self, new_tokens: &[u32], new_frames: &[usize]) -> Vec<(String, f32)> {
        self.buf_tokens.extend_from_slice(new_tokens);
        self.buf_frames.extend_from_slice(new_frames);
        let mut words = Vec::new();
        let mut prev = String::new();
        let mut cur = String::new();
        let mut cur_start = 0.0f32;
        let mut completed_upto = 0usize;
        for i in 0..self.buf_tokens.len() {
            let full = self.tok.detokenize(&self.buf_tokens[0..=i]).unwrap_or_default();
            let added = if full.len() >= prev.len() { full[prev.len()..].to_string() } else { full.clone() };
            let t = self.buf_frames[i] as f32 * FRAME_MS;
            let starts = added.starts_with(' ') || (cur.trim().is_empty() && !added.trim().is_empty());
            if starts && !cur.trim().is_empty() {
                words.push((cur.trim().to_string(), cur_start));
                completed_upto = i;
                cur.clear();
            }
            if cur.trim().is_empty() {
                cur_start = t;
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

/// A word emitted by a stream worker, on the common (offset + frame-time) axis.
enum Msg {
    Ref(String, f32),
    Mic(String, f32),
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
    let sock = UdpSocket::bind(&listen).with_context(|| format!("binding {listen}"))?;
    let mut pipe = build_pipeline(&st, &device, DType::F32, batch)?;
    let mut grouper = WordGrouper::new(Tokenizer::from_file(&tok)?);
    let mut start: Option<Instant> = None;
    let mut last_recv: Option<Instant> = None;
    let mut pushed: usize = 0;
    let mut buf = vec![0u8; 65_536];
    loop {
        let n = match sock.recv(&mut buf) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let samples = parse_l16(&buf[..n]);
        if samples.is_empty() {
            continue;
        }
        let now = Instant::now();
        let s0 = *start.get_or_insert(now);
        let off = s0.duration_since(prog_start).as_millis() as f32;
        let _ = (&mut last_recv, &mut pushed);
        // NOTE: word abs time = stream-start offset + pipeline frame-time. This
        // is accurate for the continuous mic but DRIFTS for the reference across
        // its TTS silence gaps (frame-time counts only received audio). Wall-clock
        // silence-padding to compensate proved fragile — once the streaming path
        // falls behind real-time, wall-clock-derived timing inflates. The correct
        // fix is an RTP-timestamp-based audio clock (immune to processing lag) +
        // cross-stream sync; left for a focused pass. See handoff "tag_live timing".
        pipe.push_audio(&samples);
        pushed += samples.len();
        while let Some(toks) = pipe.try_advance()? {
            if toks.is_empty() {
                continue;
            }
            let prev = pipe.all_tokens.len() - toks.len();
            let frames = pipe.all_frames[prev..].to_vec();
            for (w, t) in grouper.push(&toks, &frames) {
                let msg = if is_mic { Msg::Mic(w, off + t) } else { Msg::Ref(w, off + t) };
                if tx.send(msg).is_err() {
                    return Ok(());
                }
            }
        }
    }
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
    let mut ref_latest_ms: f32 = 0.0;
    let mut last_ref_word = Instant::now();
    let hold = Duration::from_millis(args.hold_ms);

    loop {
        match rx.recv_timeout(Duration::from_millis(40)) {
            Ok(Msg::Ref(w, abs)) => {
                println!("{DIM}{RED}[ref ]{RST}{RED} {:>7.1}s  {w}{RST}", abs / 1000.0);
                ref_latest_ms = ref_latest_ms.max(abs);
                last_ref_word = Instant::now();
                ref_recent.push_back((w, abs));
            }
            Ok(Msg::Mic(w, abs)) => pending_mic.push_back((w, abs, Instant::now())),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        let now = Instant::now();
        let tts_active = now.duration_since(last_ref_word).as_millis() < 600;
        while let Some((w, t, arrived)) = pending_mic.front().cloned() {
            let ref_caught_up = ref_latest_ms >= t - args.delay_lo_ms as f32;
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
            println!(
                "{col}{BOLD}[mic ]{RST}{col} {:>7.1}s  {w:<14} {tag}{RST} {DIM}(ref~'{}' {:.2}){RST}",
                t / 1000.0, best.0, best.1
            );
            if !is_echo {
                if let (Some(sock), Some(addr)) = (&out_sock, &args.text_out) {
                    let _ = sock.send_to(format!("{w} ").as_bytes(), addr);
                }
            }
        }

        let horizon = (args.delay_hi_ms + 4000) as f32;
        while let Some((_, rt)) = ref_recent.front() {
            if ref_latest_ms - rt > horizon {
                ref_recent.pop_front();
            } else {
                break;
            }
        }
    }
    Ok(())
}
