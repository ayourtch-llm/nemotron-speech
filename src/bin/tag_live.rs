//! tag_live — dual-stream speaker-attributed ASR for the Nabu voice loop.
//!
//! Transcribes BOTH RTP streams at once on the same engine/clock:
//!   - the MIC   (near end: user + residual TTS echo)   on --mic-listen
//!   - the TTS REFERENCE (the echo's source)            on --ref-listen
//! then tags each mic word USER vs ECHO by matching it against the reference
//! words in TIME (echo lags the reference by ~delay) AND CONTENT (fuzzy). The
//! live view prints both ASR streams color-coded + the per-word tag, and only
//! USER words are forwarded to the agent (--text-out).
//!
//! The reference stream is the EASY case (clean audio) with a relaxed latency
//! budget (its echo doesn't reach the mic until ~delay later), so it runs at a
//! larger chunk-batch than the mic.

use std::collections::VecDeque;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use tokio::net::UdpSocket;

use nemotron_speech::audio_source::{AudioSource, UdpSource};
use nemotron_speech::features::{IncrementalMelExtractor, MelConfig};
use nemotron_speech::model::encoder::FastConformerEncoder;
use nemotron_speech::model::joint::JointNet;
use nemotron_speech::model::predict::PredictNet;
use nemotron_speech::model::ModelConfig;
use nemotron_speech::streaming::StreamingPipeline;
use nemotron_speech::tokenizer::Tokenizer;

const FRAME_MS: f32 = 80.0; // encoder stride: mel hop 10ms × subsample 8
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
    /// Near-end mic RTP/L16 (point rtp-aec --asr-out2 here).
    #[arg(long, default_value = "0.0.0.0:9992")]
    mic_listen: String,
    /// TTS reference RTP/L16 (point rtp-aec --ref-out2 here).
    #[arg(long, default_value = "0.0.0.0:9993")]
    ref_listen: String,
    /// Forward USER-tagged words to this UDP target (the agent). Unset = print only.
    #[arg(long)]
    text_out: Option<String>,
    #[arg(long, default_value_t = false)]
    cpu: bool,
    /// Encoder chunk fusion for the mic (low latency = small).
    #[arg(long, default_value_t = 2)]
    chunk_batch: usize,
    /// Encoder chunk fusion for the reference (relaxed latency = large).
    #[arg(long, default_value_t = 8)]
    ref_chunk_batch: usize,
    /// Echo lags the reference by [delay_lo, delay_hi] ms (the match window).
    #[arg(long, default_value_t = 50)]
    delay_lo_ms: i64,
    #[arg(long, default_value_t = 950)]
    delay_hi_ms: i64,
    #[arg(long, default_value_t = 0.55)]
    fuzzy_thr: f32,
}

fn pick_device(cpu: bool) -> Device {
    if cpu {
        return Device::Cpu;
    }
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
}

fn build_pipeline(st: &std::path::Path, device: &Device, dtype: DType) -> Result<StreamingPipeline> {
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
    StreamingPipeline::new(encoder, predict, joint, mel, mel_cfg, cfg, device.clone(), dtype)
}

/// Groups a stream's tokens into words with start timestamps. Holds only the
/// in-progress word's tokens (emits a word when the next word starts).
struct WordGrouper {
    tok: Tokenizer,
    buf_tokens: Vec<u32>,
    buf_frames: Vec<usize>,
}

impl WordGrouper {
    fn new(tok: Tokenizer) -> Self {
        Self { tok, buf_tokens: Vec::new(), buf_frames: Vec::new() }
    }

    /// Feed newly decoded (token, frame) pairs; return any words that completed.
    fn push(&mut self, new_tokens: &[u32], new_frames: &[usize]) -> Vec<(String, f32)> {
        self.buf_tokens.extend_from_slice(new_tokens);
        self.buf_frames.extend_from_slice(new_frames);
        let mut words = Vec::new();
        // Re-derive word boundaries over the (small) buffer via prefix-decode.
        let mut prev = String::new();
        let mut cur = String::new();
        let mut cur_start = 0.0f32;
        let mut completed_upto = 0usize; // token index where the last completed word ended
        for i in 0..self.buf_tokens.len() {
            let full = self.tok.detokenize(&self.buf_tokens[0..=i]).unwrap_or_default();
            let added = if full.len() >= prev.len() { full[prev.len()..].to_string() } else { full.clone() };
            let t = self.buf_frames[i] as f32 * FRAME_MS;
            let starts = added.starts_with(' ') || (cur.trim().is_empty() && !added.trim().is_empty());
            if starts && !cur.trim().is_empty() {
                words.push((cur.trim().to_string(), cur_start));
                completed_upto = i; // the completed word ends here; drain up to the new word's first token
                cur.clear();
            }
            if cur.trim().is_empty() {
                cur_start = t;
            }
            cur.push_str(&added);
            prev = full;
        }
        // Drop tokens of fully-completed words; keep the in-progress word.
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

/// Fuzzy content match (same rules as tools/echo_match.py).
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
        return 0.9; // tokenization merge/split across the two ASR passes
    }
    if lo.contains(sh.as_str()) {
        return sh.len() as f32 / lo.len() as f32;
    }
    let n = a.len().min(b.len());
    let shared = (0..n).filter(|&i| a.as_bytes()[i] == b.as_bytes()[i]).count();
    shared as f32 / a.len().max(b.len()) as f32
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let device = pick_device(args.cpu);
    let dtype = DType::F32;
    eprintln!("tag_live: device {:?}", device);

    let mut mic_pipe = build_pipeline(&args.st, &device, dtype)?;
    mic_pipe.set_max_chunk_batch(args.chunk_batch);
    let mut ref_pipe = build_pipeline(&args.st, &device, dtype)?;
    ref_pipe.set_max_chunk_batch(args.ref_chunk_batch);
    let mut mic_words = WordGrouper::new(Tokenizer::from_file(&args.tok)?);
    let mut ref_words = WordGrouper::new(Tokenizer::from_file(&args.tok)?);
    eprintln!("tag_live: loaded 2 pipelines (mic batch {}, ref batch {})", args.chunk_batch, args.ref_chunk_batch);

    let mut mic_src = UdpSource::bind(&args.mic_listen, 320).await?;
    let mut ref_src = UdpSource::bind(&args.ref_listen, 320).await?;
    eprintln!("tag_live: mic {} | ref {}", args.mic_listen, args.ref_listen);

    let out_sock = match &args.text_out {
        Some(_) => Some(UdpSocket::bind("0.0.0.0:0").await?),
        None => None,
    };

    // Recent reference words (audio time), the in-flight mic words awaiting a
    // verdict, the newest reference audio-time seen, and when reference audio
    // last flowed (to tell "TTS silent" from "reference ASR just lagging").
    let mut ref_recent: VecDeque<(String, f32)> = VecDeque::new();
    let mut pending_mic: VecDeque<(String, f32, std::time::Instant)> = VecDeque::new();
    let mut ref_latest_ms: f32 = 0.0;
    let mut last_ref_audio: Option<std::time::Instant> = None;
    // Common time axis: a word's abs time = (its stream's start offset) +
    // frame-time. To keep frame-time == wall-clock through the reference's
    // silence gaps (TTS pauses send no packets), each stream is padded with
    // zeros so its total pushed samples track real-time elapsed. Then both
    // streams' abs times are comparable and the echo lands ~delay after its ref.
    let prog_start = std::time::Instant::now();
    let mut mic_start: Option<std::time::Instant> = None;
    let mut ref_start: Option<std::time::Instant> = None;
    let mut mic_pushed: usize = 0;
    let mut ref_pushed: usize = 0;
    // Max wall-clock a mic word waits for the reference to catch up before we
    // commit its tag (covers the no-echo case where the reference never advances).
    let hold = std::time::Duration::from_millis(1500);
    let mut drain_timer = tokio::time::interval(std::time::Duration::from_millis(40));
    drain_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    eprintln!("tag_live: live. {GREEN}USER{RST} / {RED}ECHO{RST}\n");

    loop {
        tokio::select! {
            r = ref_src.next_chunk() => {
                let Some(chunk) = r? else { break };
                let now = std::time::Instant::now();
                last_ref_audio = Some(now);
                let rs = *ref_start.get_or_insert(now);
                let off = rs.duration_since(prog_start).as_millis() as f32;
                // Pad silence so pushed samples track real-time (handles TTS gaps).
                let target = now.duration_since(rs).as_millis() as usize * 16;
                if target > ref_pushed + chunk.samples.len() {
                    let pad = target - ref_pushed - chunk.samples.len();
                    ref_pipe.push_audio(&vec![0.0f32; pad]);
                    ref_pushed += pad;
                }
                ref_pipe.push_audio(&chunk.samples);
                ref_pushed += chunk.samples.len();
                while let Some(toks) = ref_pipe.try_advance()? {
                    if toks.is_empty() { continue; }
                    let prev = ref_pipe.all_tokens.len() - toks.len();
                    let frames = ref_pipe.all_frames[prev..].to_vec();
                    for (w, t) in ref_words.push(&toks, &frames) {
                        let abs = off + t;
                        println!("{DIM}{RED}[ref ]{RST}{RED} {:>7.1}s  {w}{RST}", abs / 1000.0);
                        ref_latest_ms = ref_latest_ms.max(abs);
                        ref_recent.push_back((w, abs));
                    }
                }
            }
            m = mic_src.next_chunk() => {
                let Some(chunk) = m? else { break };
                let now = std::time::Instant::now();
                let ms = *mic_start.get_or_insert(now);
                let off = ms.duration_since(prog_start).as_millis() as f32;
                let target = now.duration_since(ms).as_millis() as usize * 16;
                if target > mic_pushed + chunk.samples.len() {
                    let pad = target - mic_pushed - chunk.samples.len();
                    mic_pipe.push_audio(&vec![0.0f32; pad]);
                    mic_pushed += pad;
                }
                mic_pipe.push_audio(&chunk.samples);
                mic_pushed += chunk.samples.len();
                while let Some(toks) = mic_pipe.try_advance()? {
                    if toks.is_empty() { continue; }
                    let prev = mic_pipe.all_tokens.len() - toks.len();
                    let frames = mic_pipe.all_frames[prev..].to_vec();
                    for (w, t) in mic_words.push(&toks, &frames) {
                        pending_mic.push_back((w, off + t, now));
                    }
                }
            }
            _ = drain_timer.tick() => {}
        }

        // Commit any pending mic words whose verdict is now decidable: the
        // reference has advanced past the latest possible echo source for that
        // word, OR TTS is silent, OR the wait timed out.
        let now = std::time::Instant::now();
        let tts_active = last_ref_audio.map(|i| now.duration_since(i).as_millis() < 300).unwrap_or(false);
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
                    let _ = sock.send_to(format!("{w} ").as_bytes(), addr).await;
                }
            }
        }

        // Prune reference words older than the match window needs.
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
