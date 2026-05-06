//! Capture audio from the default microphone, resample to 16 kHz mono
//! f32, and send raw PCM datagrams to a UDP target. Intended as a clean
//! reusable example of the wire format that `transcribe_live
//! --udp-listen` accepts — and as a software stand-in for the future
//! hardware sender gadget.
//!
//! Build: `cargo build --release --features mic --bin udp_mic_send`
//! Run:   `./target/release/udp_mic_send --target 127.0.0.1:9999`
//!
//! Companion: `transcribe_live --udp-listen 0.0.0.0:9999` on the receive side.

#![cfg(feature = "mic")]

use anyhow::{Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::net::{ToSocketAddrs, UdpSocket};

#[derive(Parser, Debug)]
struct Args {
    /// Target host:port for UDP packets (raw f32-LE 16kHz mono PCM).
    #[arg(long)]
    target: String,
    /// Samples per datagram. 320 = 20 ms at 16 kHz; small enough for any MTU.
    #[arg(long, default_value_t = 320)]
    chunk_samples: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    let socket = UdpSocket::bind("0.0.0.0:0").context("bind ephemeral UDP socket")?;
    // Resolve once. We use send_to (not connect+send) so a transient
    // "port unreachable" from the kernel doesn't latch a permanent error
    // on this socket — useful when the receiver starts up after we do.
    let target_addr: std::net::SocketAddr = args
        .target
        .to_socket_addrs()
        .context("resolving target")?
        .next()
        .ok_or_else(|| anyhow::anyhow!("no addresses for {}", args.target))?;
    eprintln!("sending to {} from {}", target_addr, socket.local_addr()?);

    // mpsc buffer: depth tuned high enough that the cpal callback never
    // backpressures. At 320 samples / packet that's ~80 seconds of audio
    // — far more than any pipeline pause.
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(4096);

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no input device"))?;
    let config = device.default_input_config()?;
    let src_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    eprintln!(
        "mic: {} Hz, {} channel(s), {:?}",
        src_rate,
        channels,
        config.sample_format()
    );

    let target_rate = 16_000u32;
    let err_fn = |e| tracing::warn!("mic stream error: {e:?}");
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            let tx = tx.clone();
            device.build_input_stream(
                &config.config(),
                move |data: &[f32], _: &_| {
                    let mono: Vec<f32> = if channels == 1 {
                        data.to_vec()
                    } else {
                        data.chunks(channels)
                            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                            .collect()
                    };
                    let resampled = resample_linear(&mono, src_rate, target_rate);
                    let _ = tx.try_send(resampled);
                },
                err_fn,
                None,
            )?
        }
        fmt => anyhow::bail!("unsupported mic sample format {:?}", fmt),
    };
    stream.play()?;

    // Drain the channel into fixed-size UDP datagrams.
    let chunk_samples = args.chunk_samples;
    let mut buf: Vec<f32> = Vec::with_capacity(chunk_samples * 4);
    let mut bytes = vec![0u8; chunk_samples * 4];
    let mut datagrams: u64 = 0;
    let started = std::time::Instant::now();
    let mut last_log = started;
    loop {
        let samples = rx.recv()?; // blocks until next callback
        buf.extend_from_slice(&samples);
        while buf.len() >= chunk_samples {
            for i in 0..chunk_samples {
                let le = buf[i].to_le_bytes();
                bytes[i * 4..i * 4 + 4].copy_from_slice(&le);
            }
            // Tolerate transient "port unreachable" while the receiver
            // is still warming up — log and keep going.
            if let Err(e) = socket.send_to(&bytes, target_addr) {
                tracing::debug!("udp send: {e}");
            }
            buf.drain(..chunk_samples);
            datagrams += 1;
        }
        let now = std::time::Instant::now();
        if now.duration_since(last_log).as_secs() >= 5 {
            let elapsed = now.duration_since(started).as_secs_f32();
            eprintln!(
                "sent {} datagrams in {:.1}s ({:.0} pkt/s)",
                datagrams,
                elapsed,
                datagrams as f32 / elapsed
            );
            last_log = now;
        }
    }
}

/// Same per-callback linear resampler used by `MicSource`. Each callback
/// is resampled independently — adequate for testing; a streaming
/// resampler with sub-sample state would be more accurate.
fn resample_linear(x: &[f32], src: u32, dst: u32) -> Vec<f32> {
    if src == dst {
        return x.to_vec();
    }
    let step = src as f64 / dst as f64;
    let out_len = ((x.len() as f64) * dst as f64 / src as f64).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let s = i as f64 * step;
        let lo = s.floor() as usize;
        let f = (s - lo as f64) as f32;
        let s0 = x[lo.min(x.len().saturating_sub(1))];
        let s1 = x[(lo + 1).min(x.len().saturating_sub(1))];
        out.push(s0 + (s1 - s0) * f);
    }
    out
}
