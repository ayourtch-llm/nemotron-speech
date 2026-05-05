//! End-to-end streaming pipeline: audio samples in, transcribed token IDs out.
//!
//! The encoder layers run with KV + conv caches per chunk and produce
//! bit-equivalent output to the offline path. The audio→mel→subsample
//! front-end currently re-runs over the full accumulated audio buffer
//! whenever a new chunk advances; this is correct (the subsampling stack
//! is causal in time) but does redundant work that grows with utterance
//! length. A truly incremental front-end is straightforward future work
//! (mel needs preemph + reflect-pad state; subsample needs a small
//! per-stage conv state cache).

use crate::features::{MelConfig, MelExtractor};
use crate::model::ModelConfig;
use crate::model::encoder::{EncoderCache, FastConformerEncoder};
use crate::model::greedy::{GreedyDecoder, GreedyDecoderConfig};
use crate::model::joint::JointNet;
use crate::model::predict::PredictNet;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub struct StreamingPipeline {
    pub encoder: FastConformerEncoder,
    pub predict: PredictNet,
    pub joint: JointNet,
    pub mel: MelExtractor,
    pub mel_cfg: MelConfig,
    pub cfg: ModelConfig,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,

    // Live state.
    audio_buf: Vec<f32>,
    encoded_so_far: usize,        // number of encoded frames already pushed through the encoder
    cache: EncoderCache,
    decoder: GreedyDecoder,
    pub all_tokens: Vec<u32>,
    finished: bool,
}

impl StreamingPipeline {
    pub fn new(
        encoder: FastConformerEncoder,
        predict: PredictNet,
        joint: JointNet,
        mel: MelExtractor,
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
            audio_buf: Vec::with_capacity(16_000 * 30),
            encoded_so_far: 0,
            cache,
            decoder,
            all_tokens: Vec::new(),
            finished: false,
        })
    }

    /// Append more audio. Does no model work; just buffers samples.
    pub fn push_audio(&mut self, samples: &[f32]) {
        self.audio_buf.extend_from_slice(samples);
    }

    /// Mark the input stream as ended. Subsequent advances are allowed to
    /// emit any tail content (for now we just accept the existing buffer).
    pub fn finish(&mut self) {
        self.finished = true;
    }

    /// How many encoded frames are *now* available given the audio buffered
    /// so far? Computed cheaply from buffer length (matches NeMo's
    /// `calc_length` for the dw_striding stack).
    fn available_encoded_frames(&self) -> usize {
        // mel frames: ceil((len + n_fft) / hop) - we use the same formula as
        // MelExtractor::n_frames.
        let n_mel = self.mel.n_frames(self.audio_buf.len());
        // subsample reduces by a factor of 8 with causal padding; the
        // sequence length is ceil(n_mel / 2) per stage, three stages.
        let mut k = n_mel;
        for _ in 0..3 {
            k = k / 2 + 1;
        }
        // Edge case: if there's no input at all, k should be zero.
        if n_mel == 0 {
            0
        } else {
            k
        }
    }

    /// Try to process one chunk. Returns the new tokens emitted (if any).
    /// Returns Ok(None) when there isn't enough buffered audio yet.
    pub fn try_advance(&mut self) -> Result<Option<Vec<u32>>> {
        let chunk_size = self.cfg.chunk_size_enc_frames();
        let avail = self.available_encoded_frames();
        let needed = self.encoded_so_far + chunk_size;
        // We need a *full* chunk of new frames, otherwise streaming math
        // (positional embedding lengths, KV cache size) gets uglier. On
        // finish(), accept a smaller tail chunk.
        if avail < needed {
            if self.finished && avail > self.encoded_so_far {
                // emit a partial tail chunk
                return self.advance_chunk(avail - self.encoded_so_far).map(Some);
            }
            return Ok(None);
        }
        self.advance_chunk(chunk_size).map(Some)
    }

    fn advance_chunk(&mut self, len: usize) -> Result<Vec<u32>> {
        // Recompute mel + subsample on the full audio buffer. Causal so
        // results for "old" frames are stable across re-computations.
        let log_mel = self.mel.forward(&self.audio_buf);
        let n_mel = log_mel.len() / self.mel_cfg.n_mels;
        let mel_t = Tensor::from_vec(log_mel, (1, self.mel_cfg.n_mels, n_mel), &self.device)?;

        let subsampled = self
            .encoder
            .subsample
            .forward(&mel_t)
            .map_err(|e| anyhow::anyhow!("subsample: {e:#}"))?;
        let chunk = subsampled
            .narrow(1, self.encoded_so_far, len)?
            .contiguous()?;

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
