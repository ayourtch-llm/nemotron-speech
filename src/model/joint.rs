//! RNN-T joint network. Split-projection form:
//!
//!     enc_proj = Linear(d_enc, joint_hidden)(f)
//!     pred_proj = Linear(d_pred, joint_hidden)(g)
//!     logits = Linear(joint_hidden, vocab + 1)(activation(enc_proj + pred_proj))
//!
//! Activation is ReLU per the model config. We omit the trailing log_softmax
//! since greedy argmax is invariant under it.

use crate::model::ModelConfig;
use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder};

pub struct JointNet {
    enc: Linear,
    pred: Linear,
    out: Linear,
}

impl JointNet {
    pub fn new(vb: VarBuilder, cfg: &ModelConfig) -> Result<Self> {
        let enc = {
            let w = vb
                .get((cfg.joint_hidden, cfg.d_model), "enc.weight")
                .context("joint enc weight")?;
            let b = vb
                .get(cfg.joint_hidden, "enc.bias")
                .context("joint enc bias")?;
            Linear::new(w, Some(b))
        };
        let pred = {
            let w = vb
                .get((cfg.joint_hidden, cfg.pred_hidden), "pred.weight")
                .context("joint pred weight")?;
            let b = vb
                .get(cfg.joint_hidden, "pred.bias")
                .context("joint pred bias")?;
            Linear::new(w, Some(b))
        };
        let out = {
            let w = vb
                .get((cfg.vocab_size + 1, cfg.joint_hidden), "out.weight")
                .context("joint out weight")?;
            let b = vb
                .get(cfg.vocab_size + 1, "out.bias")
                .context("joint out bias")?;
            Linear::new(w, Some(b))
        };
        Ok(Self { enc, pred, out })
    }

    /// Single (encoder frame, predictor output) pair -> logits over (vocab + 1).
    /// Both inputs are 2D `(B, D)`; output is `(B, vocab + 1)`.
    pub fn step(&self, f: &Tensor, g: &Tensor) -> Result<Tensor> {
        let f_proj = self.enc.forward(f)?;
        let g_proj = self.pred.forward(g)?;
        let h = (f_proj + g_proj)?.relu()?;
        Ok(self.out.forward(&h)?)
    }
}
