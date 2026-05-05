//! End-to-end streaming pipeline: audio samples in, transcribed token IDs out.
//!
//! The encoder layers run with KV + conv caches per chunk and produce
//! bit-equivalent output to the offline path. Mel features are computed
//! incrementally via `IncrementalMelExtractor` (preemph history +
//! reflection state), so per-chunk mel cost is O(chunk_samples) rather
//! than O(total_audio). The subsampling stack still re-runs on the full
//! mel buffer each chunk; an incremental version is the next step.

use crate::features::{IncrementalMelExtractor, MelConfig};
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
    pub mel: IncrementalMelExtractor,
    pub mel_cfg: MelConfig,
    pub cfg: ModelConfig,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,

    encoded_so_far: usize,        // number of encoded frames already pushed through the encoder
    cache: EncoderCache,
    decoder: GreedyDecoder,
    pub all_tokens: Vec<u32>,
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
            encoded_so_far: 0,
            cache,
            decoder,
            all_tokens: Vec::new(),
        })
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

    /// How many encoded frames are *now* available given the audio buffered
    /// so far? Derived from the incremental mel's frame count + the subsample
    /// stack's `floor(N/2)+1` per-stage rule (3 stages).
    fn available_encoded_frames(&self) -> usize {
        let n_mel = self.mel.available_frames();
        if n_mel == 0 {
            return 0;
        }
        let mut k = n_mel;
        for _ in 0..3 {
            k = k / 2 + 1;
        }
        k
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
            if self.mel.is_finished() && avail > self.encoded_so_far {
                return self.advance_chunk(avail - self.encoded_so_far).map(Some);
            }
            return Ok(None);
        }
        self.advance_chunk(chunk_size).map(Some)
    }

    fn advance_chunk(&mut self, len: usize) -> Result<Vec<u32>> {
        // Pull current mel from the incremental extractor — already in
        // (n_mels, T) row-major. Subsample still runs on the full buffer
        // each call (causal, so old frames are stable).
        let log_mel = self.mel.mel_buffer();
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
