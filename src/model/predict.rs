//! RNN-T prediction network: embedding + 2-layer LSTM stack.
//!
//! Vocab + 1 (blank, used as pad index, blank_as_pad=True). At greedy step
//! 0 we feed a zero embedding (SOS = blank) — NeMo's convention from
//! `decoder.predict(None, ...)`.

use crate::model::ModelConfig;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{LSTMConfig, RNN, VarBuilder, lstm};

pub struct PredictNet {
    embed: Tensor,              // (vocab_size + 1, pred_hidden)
    lstm: Vec<candle_nn::LSTM>, // length = pred_rnn_layers
    pred_hidden: usize,
    device: Device,
    dtype: DType,
}

#[derive(Clone)]
pub struct PredictState {
    pub h: Vec<Tensor>, // per-layer (B, pred_hidden)
    pub c: Vec<Tensor>,
}

impl PredictState {
    pub fn zero(
        layers: usize,
        batch: usize,
        hidden: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut h = Vec::with_capacity(layers);
        let mut c = Vec::with_capacity(layers);
        for _ in 0..layers {
            h.push(Tensor::zeros((batch, hidden), dtype, device)?.contiguous()?);
            c.push(Tensor::zeros((batch, hidden), dtype, device)?.contiguous()?);
        }
        Ok(Self { h, c })
    }
}

impl PredictNet {
    pub fn new(vb: VarBuilder, cfg: &ModelConfig) -> Result<Self> {
        let embed = vb
            .get((cfg.vocab_size + 1, cfg.pred_hidden), "embed.weight")
            .context("embed weight")?;
        let mut lstm_layers = Vec::with_capacity(cfg.pred_rnn_layers);
        for i in 0..cfg.pred_rnn_layers {
            let in_dim = if i == 0 {
                cfg.pred_hidden
            } else {
                cfg.pred_hidden
            };
            let cfg_l = LSTMConfig {
                layer_idx: i,
                ..LSTMConfig::default()
            };
            let layer =
                lstm(in_dim, cfg.pred_hidden, cfg_l, vb.pp("lstm")).context("lstm layer")?;
            lstm_layers.push(layer);
        }
        Ok(Self {
            embed,
            lstm: lstm_layers,
            pred_hidden: cfg.pred_hidden,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn pred_hidden(&self) -> usize {
        self.pred_hidden
    }

    /// Single-step forward at greedy decoding time. `last_token` is `None`
    /// at the very start (SOS = zero embedding), or `Some(idx)` to look up
    /// the embedding for the previously emitted non-blank token.
    pub fn step(
        &self,
        last_token: Option<usize>,
        state: &PredictState,
    ) -> Result<(Tensor, PredictState)> {
        let batch = 1usize;
        // Build input: (B, pred_hidden)
        let x = match last_token {
            None => {
                Tensor::zeros((batch, self.pred_hidden), self.dtype, &self.device)?.contiguous()?
            }
            Some(idx) => {
                // gather row idx from embed -> (B, pred_hidden)
                let idx_t = Tensor::from_vec(vec![idx as u32], (batch,), &self.device)?;
                self.embed.index_select(&idx_t, 0)?.contiguous()?
            }
        };

        let mut h_new = Vec::with_capacity(self.lstm.len());
        let mut c_new = Vec::with_capacity(self.lstm.len());
        let mut hin = x;
        for (i, layer) in self.lstm.iter().enumerate() {
            let in_state = candle_nn::rnn::LSTMState::new(state.h[i].clone(), state.c[i].clone());
            let new_state = layer.step(&hin, &in_state)?;
            hin = new_state.h().clone();
            h_new.push(new_state.h().clone());
            c_new.push(new_state.c().clone());
        }
        // Output of top layer: (B, pred_hidden)
        Ok((hin, PredictState { h: h_new, c: c_new }))
    }
}
