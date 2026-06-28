//! End-to-end streaming pipeline: audio samples in, transcribed token IDs out.
//!
//! Both the front-end (mel + subsample) and the encoder layers run
//! incrementally — per-chunk cost is O(chunk_size), not O(total_audio).
//! `IncrementalMelExtractor` carries preemph + reflection state;
//! `SubsampleStreamingState` carries 2-frame caches at each stride-2
//! conv stage; the conformer layers carry KV + conv-module caches.
//! Cumulative output is byte-equivalent to the offline path on `finish()`.

use crate::features::{IncrementalMelExtractor, MelConfig};
use crate::model::ModelConfig;
use crate::model::encoder::{EncoderCache, FastConformerEncoder, SubsampleStreamingState};
use crate::model::greedy::{GreedyDecoder, GreedyDecoderConfig};
use crate::model::joint::JointNet;
use crate::model::predict::PredictNet;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub struct StreamingPipeline {
    pub encoder: FastConformerEncoder,
    pub predict: PredictNet,
    pub joint: JointNet,
    pub mel: IncrementalMelExtractor,
    pub mel_cfg: MelConfig,
    pub cfg: ModelConfig,
    device: Device,
    dtype: DType,

    /// Number of mel frames already fed to the streaming subsample stack.
    mel_consumed: usize,
    sub_state: SubsampleStreamingState,
    encoded_so_far: usize, // encoded frames already pushed through the conformer encoder
    cache: EncoderCache,
    decoder: GreedyDecoder,
    pub all_tokens: Vec<u32>,
    /// Maximum number of `chunk_size` blocks to run through the conformer
    /// encoder in a single batched pass. 1 = original per-chunk behaviour
    /// (lowest latency). Larger values amortise per-op dispatch overhead —
    /// the difference between Metal being unusable (~0.8x realtime at
    /// batch 1) and keeping up — at the cost of up to `max_chunk_batch`
    /// chunks (~1.1s each) of extra algorithmic latency. Numerically
    /// identical regardless of the value (a block-causal mask reproduces
    /// the per-chunk attention pattern).
    max_chunk_batch: usize,
}

impl StreamingPipeline {
    pub fn new(
        encoder: FastConformerEncoder,
        predict: PredictNet,
        joint: JointNet,
        mel: IncrementalMelExtractor,
        mel_cfg: MelConfig,
        cfg: ModelConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let cache = EncoderCache::empty(cfg.n_layers);
        let decoder = GreedyDecoder::new(
            &predict,
            GreedyDecoderConfig {
                blank_idx: cfg.blank_idx,
                max_symbols_per_step: 10,
            },
            &device,
            dtype,
        )?;
        Ok(Self {
            encoder,
            predict,
            joint,
            mel,
            mel_cfg,
            cfg,
            device,
            dtype,
            mel_consumed: 0,
            sub_state: SubsampleStreamingState::empty(),
            encoded_so_far: 0,
            cache,
            decoder,
            all_tokens: Vec::new(),
            max_chunk_batch: 1,
        })
    }

    /// Set how many `chunk_size` blocks may be fused into one encoder pass.
    /// Clamped to at least 1. See `max_chunk_batch`.
    pub fn set_max_chunk_batch(&mut self, n: usize) {
        self.max_chunk_batch = n.max(1);
    }

    /// Append more audio. Pre-emphasis + STFT framing run as samples arrive;
    /// no encoder work happens until `try_advance()` is called.
    pub fn push_audio(&mut self, samples: &[f32]) {
        self.mel.push_audio(samples);
    }

    /// Mark the input stream as ended. Unlocks the trailing 1–2 mel frames
    /// that depend on right-side reflection.
    pub fn finish(&mut self) {
        self.mel.finish();
    }

    /// Drain newly-available mel frames through the streaming subsample
    /// stack. Idempotent: calling twice with no new mel is a no-op (unless
    /// the stream has just been finished, in which case it flushes the
    /// trailing tentative frames).
    fn drain_subsample(&mut self) -> Result<()> {
        let total_mel = self.mel.n_frames_emitted();
        let new_mel = total_mel - self.mel_consumed;
        let mel_finished = self.mel.is_finished();
        let need_call = new_mel > 0 || (mel_finished && !self.sub_state.is_finalized());
        if !need_call {
            return Ok(());
        }
        let n_mels = self.mel_cfg.n_mels;
        let slice = self.mel.mel_buffer_since(self.mel_consumed);
        let new_tensor = Tensor::from_vec(slice, (1, n_mels, new_mel), &self.device)?;
        self.encoder
            .subsample
            .forward_incremental(
                &new_tensor,
                &mut self.sub_state,
                &self.device,
                self.dtype,
                mel_finished,
            )
            .map_err(|e| anyhow::anyhow!("subsample incremental: {e:#}"))?;
        self.mel_consumed = total_mel;
        Ok(())
    }

    /// Try to process one chunk. Returns the new tokens emitted (if any).
    /// Returns Ok(None) when there isn't enough buffered audio yet.
    pub fn try_advance(&mut self) -> Result<Option<Vec<u32>>> {
        self.drain_subsample()?;
        let chunk_size = self.cfg.chunk_size_enc_frames();
        let avail = self.sub_state.n_emitted;
        let ready = avail.saturating_sub(self.encoded_so_far);
        if ready < chunk_size {
            // Not a full chunk yet. Flush the partial tail only once the
            // stream has ended (right-context will never arrive).
            if self.mel.is_finished() && ready > 0 {
                return self.advance_chunk(ready).map(Some);
            }
            return Ok(None);
        }
        // Run as many whole chunks as are buffered, up to the batch cap, in
        // a single fused encoder pass. The block-causal mask inside the
        // encoder makes this byte-identical to processing them one at a time.
        let full_chunks = (ready / chunk_size).min(self.max_chunk_batch);
        self.advance_chunk(full_chunks * chunk_size).map(Some)
    }

    fn advance_chunk(&mut self, len: usize) -> Result<Vec<u32>> {
        let enc_buf = self
            .sub_state
            .output
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("advance_chunk before any subsample output"))?;
        let chunk = enc_buf.narrow(1, self.encoded_so_far, len)?.contiguous()?;

        let enc_out = self
            .encoder
            .forward_layers_chunked(&chunk, &mut self.cache)
            .map_err(|e| anyhow::anyhow!("encoder chunk forward: {e:#}"))?;

        let enc_seq = enc_out.squeeze(0)?;
        let prev = self.all_tokens.len();
        self.decoder
            .decode(&enc_seq, &self.predict, &self.joint, &mut self.all_tokens)?;
        let new_tokens = self.all_tokens[prev..].to_vec();
        self.encoded_so_far += len;
        Ok(new_tokens)
    }
}
