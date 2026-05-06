//! NemotronSpeech model: cache-aware FastConformer + RNN-T.
//!
//! The encoder structure mirrors NeMo's reference implementation but is
//! written from scratch on top of candle. Naming follows the renamed
//! safetensors keys produced by `tools/convert_nemo.py`.

pub mod encoder;
pub mod greedy;
pub mod joint;
pub mod predict;

use candle_core::{DType, Device};

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub n_mels: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub ff_expansion: usize,
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
    pub vocab_size: usize,
    pub blank_idx: usize,
    pub pred_hidden: usize,
    pub joint_hidden: usize,
    pub pred_rnn_layers: usize,
    pub pos_emb_max_len: usize,
    /// Default chunked-limited attention context: [left_frames, right_frames]
    /// in encoded (post-subsampling, 80 ms) frames. The supported set for
    /// the published checkpoint is [70, 0|1|6|13]; we default to [70, 13]
    /// (best WER, ~1.12 s lookahead).
    pub att_context_size: [usize; 2],
}

impl ModelConfig {
    pub fn nemotron_06b() -> Self {
        Self {
            n_mels: 128,
            d_model: 1024,
            n_layers: 24,
            n_heads: 8,
            d_head: 128, // 1024 / 8
            ff_expansion: 4,
            conv_kernel: 9,
            subsampling_factor: 8,
            subsampling_channels: 256,
            vocab_size: 1024, // BPE labels
            blank_idx: 1024,  // blank is appended at the end
            pred_hidden: 640,
            joint_hidden: 640,
            pred_rnn_layers: 2,
            pos_emb_max_len: 5000,
            att_context_size: [70, 13],
        }
    }

    pub fn chunk_size_enc_frames(&self) -> usize {
        self.att_context_size[1] + 1
    }

    pub fn left_chunks(&self) -> usize {
        self.att_context_size[0] / self.chunk_size_enc_frames()
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub device: Device,
    pub dtype: DType,
}

impl RuntimeConfig {
    pub fn cpu_f32() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
        }
    }

    /// Pick a sensible default device. With the `metal` feature enabled and a
    /// Metal device available, use Metal; otherwise fall back to CPU. CUDA is
    /// gated behind the `cuda` feature similarly.
    pub fn auto() -> candle_core::Result<Self> {
        #[cfg(feature = "metal")]
        {
            if let Ok(d) = Device::new_metal(0) {
                return Ok(Self {
                    device: d,
                    dtype: DType::F32,
                });
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Ok(d) = Device::new_cuda(0) {
                return Ok(Self {
                    device: d,
                    dtype: DType::F32,
                });
            }
        }
        Ok(Self::cpu_f32())
    }
}
