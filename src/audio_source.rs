//! Pluggable audio source abstraction.
//!
//! All sources produce a stream of 16 kHz mono f32 samples. The pipeline is
//! source-agnostic — the same chunker + encoder + decoder can consume audio
//! from a file, a microphone, or future UDP packets.
//!
//! Trait choice: the model's chunked streaming interface naturally maps to
//! "give me the next batch of N samples". We use an async trait so the mic
//! and UDP sources can yield to the runtime while waiting for new audio.

use anyhow::Result;
use async_trait::async_trait;

/// One audio chunk hand-off. `is_final = true` signals the producer has no
/// more audio; the decoder should flush whatever it can.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub is_final: bool,
}

#[async_trait]
pub trait AudioSource: Send {
    /// Sample rate the source emits. Must be 16 kHz; resampling is the
    /// source's responsibility (or its caller's, if it predates this trait).
    fn sample_rate(&self) -> u32;

    /// Yield the next chunk, or `None` when the source is exhausted. Callers
    /// should keep polling until `None` or until they choose to stop.
    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>>;
}

/// Adapter: wrap a fully-loaded `Vec<f32>` and serve it in fixed-size chunks.
/// Useful for offline file processing through the streaming pipeline.
pub struct FileChunkSource {
    samples: Vec<f32>,
    pos: usize,
    chunk_samples: usize,
}

impl FileChunkSource {
    pub fn new(samples: Vec<f32>, chunk_samples: usize) -> Self {
        assert!(chunk_samples > 0);
        Self {
            samples,
            pos: 0,
            chunk_samples,
        }
    }
}

#[async_trait]
impl AudioSource for FileChunkSource {
    fn sample_rate(&self) -> u32 {
        16_000
    }

    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>> {
        if self.pos >= self.samples.len() {
            return Ok(None);
        }
        let end = (self.pos + self.chunk_samples).min(self.samples.len());
        let chunk = self.samples[self.pos..end].to_vec();
        let is_final = end == self.samples.len();
        self.pos = end;
        Ok(Some(AudioChunk {
            samples: chunk,
            is_final,
        }))
    }
}

/// UDP source: receives RTP/L16 datagrams (16 kHz mono) over UDP, strips the
/// RTP header, decodes the L16 payload to `f32`, accumulates the samples, and
/// yields fixed-size audio chunks whenever the buffer reaches `chunk_samples`.
///
/// Wire format: RTP (RFC 3550) header (12 bytes for our sender — V=2, no
/// CSRC/extension) followed by an L16 payload (RFC 3551): 16-bit signed PCM in
/// big-endian / network byte order. Non-RTP datagrams are dropped with a
/// warning. (Was raw f32-LE before the RTP migration.)
///
/// Loss/reorder: UDP gives no guarantees, but for a local LAN with a
/// single sender both are essentially zero. We don't sequence-number for
/// now — packet loss would manifest as audio gaps which the encoder
/// handles gracefully (state.md gotcha #9: subsample is causal in time,
/// old encoded frames stay stable).
pub struct UdpSource {
    socket: tokio::net::UdpSocket,
    chunk_samples: usize,
    buffer: Vec<f32>,
    recv_buf: Vec<u8>,
}

impl UdpSource {
    pub async fn bind(addr: &str, chunk_samples: usize) -> Result<Self> {
        assert!(chunk_samples > 0);
        let socket = tokio::net::UdpSocket::bind(addr).await?;
        Ok(Self {
            socket,
            chunk_samples,
            buffer: Vec::with_capacity(chunk_samples * 4),
            // 65_536 covers any UDP datagram (max IPv4 payload).
            recv_buf: vec![0u8; 65_536],
        })
    }

    /// The bound address, useful when binding to port 0.
    pub fn local_addr(&self) -> Result<std::net::SocketAddr> {
        Ok(self.socket.local_addr()?)
    }
}

#[async_trait]
impl AudioSource for UdpSource {
    fn sample_rate(&self) -> u32 {
        16_000
    }

    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>> {
        // RTP (RFC 3550) carrying an L16 payload: 16-bit signed big-endian PCM
        // @16 kHz. 12-byte header (V=2, no CSRC/extension from our sender, but
        // we parse CC/X defensively), then the samples.
        const RTP_HDR: usize = 12;
        while self.buffer.len() < self.chunk_samples {
            let (n, _peer) = self.socket.recv_from(&mut self.recv_buf).await?;
            if n < RTP_HDR + 2 || (self.recv_buf[0] >> 6) != 2 {
                tracing::warn!("ignoring non-RTP datagram of {n} bytes (expected RTP/L16)");
                continue;
            }
            let cc = (self.recv_buf[0] & 0x0f) as usize;
            let mut hdr = RTP_HDR + cc * 4;
            if self.recv_buf[0] & 0x10 != 0 {
                // extension header: 4 bytes + (length-in-words * 4)
                if n < hdr + 4 {
                    continue;
                }
                let words =
                    ((self.recv_buf[hdr + 2] as usize) << 8) | self.recv_buf[hdr + 3] as usize;
                hdr += 4 + words * 4;
            }
            if n <= hdr {
                continue;
            }
            let n_samples = (n - hdr) / 2;
            self.buffer.reserve(n_samples);
            for i in 0..n_samples {
                let off = hdr + i * 2;
                let s = i16::from_be_bytes([self.recv_buf[off], self.recv_buf[off + 1]]);
                self.buffer.push(s as f32 / 32768.0);
            }
        }
        let chunk: Vec<f32> = self.buffer.drain(..self.chunk_samples).collect();
        // UDP streams are open-ended — the caller stops when they want.
        Ok(Some(AudioChunk {
            samples: chunk,
            is_final: false,
        }))
    }
}

/// Microphone source via cpal. Available with the `mic` feature.
///
/// cpal's `Stream` is `!Send` because most audio backends require their
/// callback to run on a specific thread (CoreAudio, WASAPI, etc.). We hide
/// that by parking the stream on a dedicated OS thread; the AudioSource
/// itself only holds the receiving end of an mpsc channel, so it stays
/// Send + Sync.
#[cfg(feature = "mic")]
pub mod mic {
    use super::*;
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    pub struct MicSource {
        rx: tokio::sync::mpsc::Receiver<Vec<f32>>,
        chunk_samples: usize,
        buffer: Vec<f32>,
        finished: bool,
        _stop_tx: std::sync::mpsc::Sender<()>, // dropping this signals the audio thread to exit
    }

    impl MicSource {
        pub fn open_default(chunk_samples: usize) -> anyhow::Result<Self> {
            let (tx, rx) = tokio::sync::mpsc::channel::<Vec<f32>>(64);
            let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
            let (ready_tx, ready_rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();

            std::thread::spawn(move || {
                let result = (|| -> anyhow::Result<cpal::Stream> {
                    let host = cpal::default_host();
                    let device = host
                        .default_input_device()
                        .ok_or_else(|| anyhow::anyhow!("no input device"))?;
                    let config = device.default_input_config()?;
                    let src_rate = config.sample_rate().0;
                    let channels = config.channels() as usize;
                    let target_rate = 16_000u32;
                    let err_fn = |e| tracing::warn!("mic stream error: {e:?}");
                    let tx = tx.clone();
                    let stream = match config.sample_format() {
                        cpal::SampleFormat::F32 => device.build_input_stream(
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
                        )?,
                        fmt => anyhow::bail!("unsupported mic sample format {:?}", fmt),
                    };
                    stream.play()?;
                    Ok(stream)
                })();
                match result {
                    Ok(stream) => {
                        let _ = ready_tx.send(Ok(()));
                        // Park the thread (and the stream) until told to stop.
                        let _ = stop_rx.recv();
                        drop(stream);
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                    }
                }
            });

            ready_rx.recv()??;
            Ok(Self {
                rx,
                chunk_samples,
                buffer: Vec::with_capacity(chunk_samples * 4),
                finished: false,
                _stop_tx: stop_tx,
            })
        }
    }

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

    #[async_trait]
    impl AudioSource for MicSource {
        fn sample_rate(&self) -> u32 {
            16_000
        }

        async fn next_chunk(&mut self) -> Result<Option<AudioChunk>> {
            if self.finished {
                return Ok(None);
            }
            while self.buffer.len() < self.chunk_samples {
                match self.rx.recv().await {
                    Some(samples) => self.buffer.extend(samples),
                    None => {
                        self.finished = true;
                        if self.buffer.is_empty() {
                            return Ok(None);
                        }
                        let chunk = std::mem::take(&mut self.buffer);
                        return Ok(Some(AudioChunk {
                            samples: chunk,
                            is_final: true,
                        }));
                    }
                }
            }
            let mut chunk: Vec<f32> = Vec::with_capacity(self.chunk_samples);
            chunk.extend(self.buffer.drain(..self.chunk_samples));
            Ok(Some(AudioChunk {
                samples: chunk,
                is_final: false,
            }))
        }
    }
}
