//! Greedy RNN-T decoding (single-stream, blank-as-pad).
//!
//! Outer loop iterates encoder frames; the inner `while` emits up to
//! `max_symbols` non-blank tokens per frame. The predictor state and last
//! emitted token persist across encoder frames AND across streaming chunks.

use crate::model::joint::JointNet;
use crate::model::predict::{PredictNet, PredictState};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub struct GreedyDecoderConfig {
    pub blank_idx: usize,
    pub max_symbols_per_step: usize,
}

impl Default for GreedyDecoderConfig {
    fn default() -> Self {
        Self {
            blank_idx: 1024,
            max_symbols_per_step: 10,
        }
    }
}

pub struct GreedyDecoder {
    pub last_token: Option<usize>,
    pub state: PredictState,
    pub blank_idx: usize,
    pub max_symbols: usize,
}

impl GreedyDecoder {
    pub fn new(
        predict: &PredictNet,
        cfg: GreedyDecoderConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let state = PredictState::zero(2, 1, predict.pred_hidden(), device, dtype)?;
        Ok(Self {
            last_token: None,
            state,
            blank_idx: cfg.blank_idx,
            max_symbols: cfg.max_symbols_per_step,
        })
    }

    /// Process one encoder output sequence `(T, d_enc)` and append decoded
    /// tokens to `out`. State is updated in place so the next chunk's call
    /// continues from where this one left off.
    pub fn decode(
        &mut self,
        encoded: &Tensor,
        predict: &PredictNet,
        joint: &JointNet,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        let (t_frames, _d) = encoded.dims2()?;
        for t in 0..t_frames {
            // f: (1, d_enc) — one encoder frame as a row.
            let f = encoded.narrow(0, t, 1)?;

            let mut symbols_added = 0usize;
            loop {
                if symbols_added >= self.max_symbols {
                    break;
                }
                let (g, new_state) = predict.step(self.last_token, &self.state)?;
                let logits = joint.step(&f, &g)?; // (1, V+1)
                let logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;
                let argmax = logits_vec
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(self.blank_idx);

                if argmax == self.blank_idx {
                    // Emit blank: do NOT advance predictor state, advance time instead.
                    break;
                }
                // Non-blank: emit, advance predictor state and last_token.
                out.push(argmax as u32);
                self.state = new_state;
                self.last_token = Some(argmax);
                symbols_added += 1;
            }
        }
        Ok(())
    }
}
