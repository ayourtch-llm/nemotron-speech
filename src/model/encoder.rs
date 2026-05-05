//! FastConformer encoder.
//!
//! Forward path (offline, no streaming caches yet):
//!
//! 1. mel `(B, n_mels, T)` -> reshape `(B, 1, T, n_mels)`
//! 2. dw_striding subsampling x8 -> `(B, T/8, d_model)`
//! 3. + scaled relative positional embedding (computed inside attention)
//! 4. 24 ConformerLayers, each:
//!     residual + 0.5 * FFN1(LN(x))
//!     residual + Attn(LN(x))
//!     residual + Conv(LN(x))
//!     residual + 0.5 * FFN2(LN(x))
//!     LN
//! 5. final per-frame d_model output `(B, T/8, d_model)`
//!
//! Streaming + caches are layered on later.

use crate::model::ModelConfig;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn linear(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    let w = vb.get((out_dim, in_dim), "weight").context("linear weight")?;
    let b = if bias {
        Some(vb.get(out_dim, "bias").context("linear bias")?)
    } else {
        None
    };
    Ok(Linear::new(w, b))
}

fn layer_norm(vb: VarBuilder, dim: usize, eps: f64) -> Result<LayerNorm> {
    let w = vb.get(dim, "weight").context("ln weight")?;
    let b = vb.get(dim, "bias").context("ln bias")?;
    Ok(LayerNorm::new(w, b, eps))
}

/// CausalConv2D pad: (left=k-1, right=s-1) on both H and W axes; then plain
/// Conv2d with padding=0 and the desired stride.
fn pad_causal2d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    // candle's Tensor::pad_with_zeros pads a single dim. Apply twice.
    let x = x
        .pad_with_zeros(D::Minus2, left, right)?
        .pad_with_zeros(D::Minus1, left, right)?;
    Ok(x)
}

fn make_conv2d(
    vb: VarBuilder,
    in_c: usize,
    out_c: usize,
    k: usize,
    stride: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv2d> {
    let w = vb
        .get((out_c, in_c / groups, k, k), "weight")
        .context("conv2d weight")?;
    let b = if bias {
        Some(vb.get(out_c, "bias").context("conv2d bias")?)
    } else {
        None
    };
    let cfg = Conv2dConfig {
        padding: 0,
        stride,
        dilation: 1,
        groups,
        cudnn_fwd_algo: None,
    };
    Ok(if let Some(bias) = b {
        Conv2d::new(w, Some(bias), cfg)
    } else {
        Conv2d::new(w, None, cfg)
    })
}

// ---------------------------------------------------------------------------
// Subsampling: dw_striding x8  -- causal padding on both axes
// ---------------------------------------------------------------------------

pub struct DwStridingSubsampling {
    conv0: Conv2d, // 1 -> 256, k=3, s=2
    dw1: Conv2d,   // 256 -> 256, k=3, s=2, groups=256
    pw1: Conv2d,   // 256 -> 256, k=1
    dw2: Conv2d,   // 256 -> 256, k=3, s=2, groups=256
    pw2: Conv2d,   // 256 -> 256, k=1
    out: Linear,   // (n_freq_after * 256) -> d_model
    n_mels: usize,
}

impl DwStridingSubsampling {
    pub fn new(vb: VarBuilder, cfg: &ModelConfig) -> Result<Self> {
        let c = cfg.subsampling_channels;
        let conv0 = make_conv2d(vb.pp("conv0"), 1, c, 3, 2, 1, true)?;
        let dw1 = make_conv2d(vb.pp("dw1"), c, c, 3, 2, c, true)?;
        let pw1 = make_conv2d(vb.pp("pw1"), c, c, 1, 1, 1, true)?;
        let dw2 = make_conv2d(vb.pp("dw2"), c, c, 3, 2, c, true)?;
        let pw2 = make_conv2d(vb.pp("pw2"), c, c, 1, 1, 1, true)?;

        // After 3 stride-2 stages on freq dim n_mels with causal pad (2, 1):
        //   f_i = floor((f_{i-1} + 3 - 3) / 2) + 1 = floor(f_{i-1} / 2) + 1
        let mut f = cfg.n_mels;
        for _ in 0..3 {
            f = f / 2 + 1;
        }
        let out = linear(vb.pp("out"), f * c, cfg.d_model, true)?;
        Ok(Self {
            conv0,
            dw1,
            pw1,
            dw2,
            pw2,
            out,
            n_mels: cfg.n_mels,
        })
    }

    /// Input: mel `(B, n_mels, T)`. Output: `(B, T_out, d_model)`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Reshape (B, n_mels, T) -> (B, 1, T, n_mels)
        let (b, m, _t) = mel.dims3()?;
        debug_assert_eq!(m, self.n_mels);
        let x = mel.transpose(1, 2)?.unsqueeze(1)?; // (B, 1, T, n_mels)

        // Stage 1: causal pad (2,1), conv0 stride 2, ReLU
        let x = pad_causal2d(&x, 2, 1)?;
        let x = self.conv0.forward(&x)?.relu()?;
        // Stage 2: depthwise + pointwise + ReLU
        let x = pad_causal2d(&x, 2, 1)?;
        let x = self.dw1.forward(&x)?;
        let x = self.pw1.forward(&x)?.relu()?;
        // Stage 3: depthwise + pointwise + ReLU
        let x = pad_causal2d(&x, 2, 1)?;
        let x = self.dw2.forward(&x)?;
        let x = self.pw2.forward(&x)?.relu()?;

        // (B, C, T', F') -> (B, T', C*F')
        let (_b, c, tp, f) = x.dims4()?;
        let x = x.permute((0, 2, 1, 3))?.contiguous()?.reshape((b, tp, c * f))?;
        let x = self.out.forward(&x)?;
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// FeedForward (Macaron half): Linear -> Swish -> Linear   (no bias on linears)
// ---------------------------------------------------------------------------

pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, d_model: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            linear1: linear(vb.pp("linear1"), d_model, d_ff, false)?,
            linear2: linear(vb.pp("linear2"), d_ff, d_model, false)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = candle_nn::ops::silu(&x)?;
        let x = self.linear2.forward(&x)?;
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Convolution module (causal depthwise conv kernel=9 with GLU)
// ---------------------------------------------------------------------------

pub struct ConvModule {
    pw1: Conv2d,    // (B, d, 1, T) -> (B, 2d, 1, T)  pointwise
    dw: Conv2d,     // depthwise k=9 along time, groups=d_model
    norm: LayerNorm, // applied per channel (named `batch_norm` in NeMo but loaded as LN here)
    pw2: Conv2d,    // (B, d, 1, T) -> (B, d, 1, T)  pointwise
    d_model: usize,
    kernel: usize,
}

impl ConvModule {
    pub fn new(vb: VarBuilder, cfg: &ModelConfig) -> Result<Self> {
        // Note: NeMo stores conv as 1d (Conv1d) with shape (out, in/groups, k).
        // For candle we use Conv2d on (B, d, 1, T) — equivalent for 1d operation.
        // We need to coerce the 1d weights into the 2d layout.
        // The encoder's `use_bias=false` applies here: pw1/dw/pw2 have NO bias.
        let d = cfg.d_model;
        let k = cfg.conv_kernel;
        // pw1: stored shape (2d, d, 1) -> coerce to (2d, d, 1, 1) for Conv2d
        let pw1_w = vb
            .get((2 * d, d, 1), "pw1.weight")
            .context("pw1 weight")?
            .reshape((2 * d, d, 1, 1))?;
        // dw: stored shape (d, 1, k) -> (d, 1, 1, k) on Conv2d, groups=d_model
        let dw_w = vb
            .get((d, 1, k), "dw.weight")
            .context("dw weight")?
            .reshape((d, 1, 1, k))?;
        // pw2: (d, d, 1) -> (d, d, 1, 1)
        let pw2_w = vb
            .get((d, d, 1), "pw2.weight")
            .context("pw2 weight")?
            .reshape((d, d, 1, 1))?;

        let pw1 = Conv2d::new(
            pw1_w,
            None,
            Conv2dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );
        let dw = Conv2d::new(
            dw_w,
            None,
            Conv2dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: d,
                cudnn_fwd_algo: None,
            },
        );
        let pw2 = Conv2d::new(
            pw2_w,
            None,
            Conv2dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );

        let norm = layer_norm(vb.pp("norm"), d, 1e-5)?;
        Ok(Self {
            pw1,
            dw,
            norm,
            pw2,
            d_model: d,
            kernel: k,
        })
    }

    /// Input: `(B, T, d_model)` (post-LN of the outer block).
    /// Output: `(B, T, d_model)`.
    /// `conv_context_size = causal` -> left=k-1, right=0 padding on time.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _t, d) = x.dims3()?;
        debug_assert_eq!(d, self.d_model);
        // (B, T, d) -> (B, d, 1, T)
        let x = x.transpose(1, 2)?.unsqueeze(2)?;
        // pointwise -> (B, 2d, 1, T)
        let x = self.pw1.forward(&x)?;
        // GLU along channel dim: split (a, b) on channel, gated = a * sigmoid(b)
        let half = self.d_model;
        let a = x.narrow(1, 0, half)?;
        let b = x.narrow(1, half, half)?;
        let x = a.broadcast_mul(&candle_nn::ops::sigmoid(&b)?)?;
        // depthwise causal conv: pad left=(k-1), right=0 on time
        let x = x.pad_with_zeros(D::Minus1, self.kernel - 1, 0)?;
        let x = self.dw.forward(&x)?;
        // norm: applied across channel dim; LayerNorm expects last-dim. Move
        // channel last: (B, d, 1, T) -> (B, T, 1, d) -> (B, T, d)
        let (_b2, dc, _h, _tt) = x.dims4()?;
        debug_assert_eq!(dc, self.d_model);
        let xn = x.permute((0, 3, 2, 1))?.squeeze(2)?; // (B, T, d)
        let xn = self.norm.forward(&xn)?;
        // Swish + back to (B, d, 1, T)
        let xn = candle_nn::ops::silu(&xn)?;
        let xn = xn.unsqueeze(2)?.permute((0, 3, 2, 1))?.contiguous()?;
        // pointwise2 -> (B, d, 1, T) -> (B, T, d)
        let xn = self.pw2.forward(&xn)?;
        let xn = xn.squeeze(2)?.transpose(1, 2)?; // (B, T, d)
        Ok(xn)
    }
}

// ---------------------------------------------------------------------------
// Multi-Head Attention with relative positional embeddings (Transformer-XL)
// ---------------------------------------------------------------------------
// We start with full-attention (no mask) to validate algebra. Cache + chunked
// masking come in a later commit.

pub struct RelPosMha {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    pos: Linear,
    pos_bias_u: Tensor, // (h, d_head)
    pos_bias_v: Tensor, // (h, d_head)
    n_heads: usize,
    d_head: usize,
}

impl RelPosMha {
    pub fn new(vb: VarBuilder, cfg: &ModelConfig) -> Result<Self> {
        let d = cfg.d_model;
        let q = linear(vb.pp("q"), d, d, false)?;
        let k = linear(vb.pp("k"), d, d, false)?;
        let v = linear(vb.pp("v"), d, d, false)?;
        let out = linear(vb.pp("out"), d, d, false)?;
        let pos = linear(vb.pp("pos"), d, d, false)?;
        let pos_bias_u = vb
            .get((cfg.n_heads, cfg.d_head), "pos_bias_u")
            .context("pos_bias_u")?;
        let pos_bias_v = vb
            .get((cfg.n_heads, cfg.d_head), "pos_bias_v")
            .context("pos_bias_v")?;
        Ok(Self {
            q,
            k,
            v,
            out,
            pos,
            pos_bias_u,
            pos_bias_v,
            n_heads: cfg.n_heads,
            d_head: cfg.d_head,
        })
    }

    /// `x`: `(B, T, d_model)`; `pos_emb`: `(1, 2T-1, d_model)` sinusoidal.
    pub fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let h = self.n_heads;
        let dh = self.d_head;

        // Project Q, K, V; reshape to (B, h, T, dh)
        let q = self
            .q
            .forward(x)?
            .reshape((b, t, h, dh))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(x)?
            .reshape((b, t, h, dh))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape((b, t, h, dh))?
            .transpose(1, 2)?
            .contiguous()?;

        // pos: (1, 2T-1, d) -> (1, h, 2T-1, dh) -> (h, 2T-1, dh) for broadcast
        let p = self
            .pos
            .forward(pos_emb)?
            .reshape((1, pos_emb.dims3()?.1, h, dh))?
            .transpose(1, 2)?
            .contiguous()?; // (1, h, 2T-1, dh)

        // Q with biases:
        //   q_u = q + pos_bias_u (per-head)
        //   q_v = q + pos_bias_v
        // pos_bias_* shape: (h, dh) -> (1, h, 1, dh) for broadcast
        let bias_u = self
            .pos_bias_u
            .reshape((1, h, 1, dh))?
            .to_dtype(x.dtype())?;
        let bias_v = self
            .pos_bias_v
            .reshape((1, h, 1, dh))?
            .to_dtype(x.dtype())?;
        let q_u = q.broadcast_add(&bias_u)?;
        let q_v = q.broadcast_add(&bias_v)?;

        // matrix_ac = q_u @ k^T  -> (B, h, T, T)
        let matrix_ac = q_u.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;

        // matrix_bd = q_v @ p^T -> shape (B, h, T, 2T-1), then rel_shift to (B, h, T, T)
        let p_kt = p.transpose(D::Minus2, D::Minus1)?.contiguous()?; // (1, h, dh, 2T-1)
        let matrix_bd_raw = q_v.broadcast_matmul(&p_kt)?; // (B, h, T, 2T-1)
        let matrix_bd = rel_shift(&matrix_bd_raw, t)?; // (B, h, T, T)

        let scale = 1.0 / (dh as f64).sqrt();
        let scores = (matrix_ac + matrix_bd)?.affine(scale, 0.0)?;

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v)?; // (B, h, T, dh)
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, h * dh))?;
        let out = self.out.forward(&ctx)?;
        Ok(out)
    }
}

/// Transformer-XL relative shift. Input `(B, h, T, 2T-1)` -> `(B, h, T, T)`.
fn rel_shift(x: &Tensor, t: usize) -> Result<Tensor> {
    let (b, h, t1, l) = x.dims4()?;
    debug_assert_eq!(t1, t);
    debug_assert_eq!(l, 2 * t - 1);
    // Pad a column of zeros at the front of the last dim: (B, h, T, 2T)
    let zero = Tensor::zeros((b, h, t, 1), x.dtype(), x.device())?;
    let x = Tensor::cat(&[&zero, x], D::Minus1)?; // (B, h, T, 2T)
    // Reshape to (B, h, 2T, T)
    let x = x.reshape((b, h, 2 * t, t))?;
    // Drop first row along the new T-major dim, keep last 2T-1 rows: (B, h, 2T-1, T)
    let x = x.narrow(D::Minus2, 1, 2 * t - 1)?;
    // Reshape back to (B, h, T, 2T-1)
    let x = x.reshape((b, h, t, 2 * t - 1))?;
    // Take left half + diagonal: keep first T columns -> (B, h, T, T)
    let x = x.narrow(D::Minus1, 0, t)?;
    Ok(x.contiguous()?)
}

// ---------------------------------------------------------------------------
// Conformer layer
// ---------------------------------------------------------------------------

pub struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_attn: LayerNorm,
    attn: RelPosMha,
    norm_conv: LayerNorm,
    conv: ConvModule,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    pub fn new(vb: VarBuilder, cfg: &ModelConfig) -> Result<Self> {
        let d = cfg.d_model;
        let d_ff = d * cfg.ff_expansion;
        Ok(Self {
            norm_ff1: layer_norm(vb.pp("norm_ff1"), d, 1e-5)?,
            ff1: FeedForward::new(vb.pp("ff1"), d, d_ff)?,
            norm_attn: layer_norm(vb.pp("norm_attn"), d, 1e-5)?,
            attn: RelPosMha::new(vb.pp("attn"), cfg)?,
            norm_conv: layer_norm(vb.pp("norm_conv"), d, 1e-5)?,
            conv: ConvModule::new(vb.pp("conv"), cfg)?,
            norm_ff2: layer_norm(vb.pp("norm_ff2"), d, 1e-5)?,
            ff2: FeedForward::new(vb.pp("ff2"), d, d_ff)?,
            norm_out: layer_norm(vb.pp("norm_out"), d, 1e-5)?,
        })
    }

    pub fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        // FF1 macaron half
        let r = x.clone();
        let y = self.norm_ff1.forward(x)?;
        let y = self.ff1.forward(&y)?.affine(0.5, 0.0)?;
        let x = (r + y)?;

        // Self-attention
        let r = x.clone();
        let y = self.norm_attn.forward(&x)?;
        let y = self.attn.forward(&y, pos_emb)?;
        let x = (r + y)?;

        // Conv
        let r = x.clone();
        let y = self.norm_conv.forward(&x)?;
        let y = self.conv.forward(&y)?;
        let x = (r + y)?;

        // FF2 macaron half
        let r = x.clone();
        let y = self.norm_ff2.forward(&x)?;
        let y = self.ff2.forward(&y)?.affine(0.5, 0.0)?;
        let x = (r + y)?;

        // Final LN
        let x = self.norm_out.forward(&x)?;
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal relative positional embedding (NeMo RelPositionalEncoding form)
// ---------------------------------------------------------------------------

pub fn rel_position_emb(
    seq_len: usize,
    d_model: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // pe of length 2*seq_len-1, indices from +seq_len-1 down to -seq_len+1.
    let total = 2 * seq_len - 1;
    let mut data = vec![0.0f32; total * d_model];
    for k in 0..total {
        let pos = (seq_len as i32 - 1 - k as i32) as f32;
        for i in 0..d_model / 2 {
            let denom = (10000f32).powf(2.0 * i as f32 / d_model as f32);
            let theta = pos / denom;
            data[k * d_model + 2 * i] = theta.sin();
            data[k * d_model + 2 * i + 1] = theta.cos();
        }
    }
    let t = Tensor::from_vec(data, (1, total, d_model), device)?.to_dtype(dtype)?;
    Ok(t)
}

// ---------------------------------------------------------------------------
// FastConformerEncoder (offline path)
// ---------------------------------------------------------------------------

pub struct FastConformerEncoder {
    pub subsample: DwStridingSubsampling,
    pub layers: Vec<ConformerLayer>,
    pub cfg: ModelConfig,
}

impl FastConformerEncoder {
    pub fn new(vb: VarBuilder, cfg: ModelConfig) -> Result<Self> {
        let subsample = DwStridingSubsampling::new(vb.pp("subsample"), &cfg)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(ConformerLayer::new(
                vb.pp(format!("layers.{i}")),
                &cfg,
            )?);
        }
        Ok(Self { subsample, layers, cfg })
    }

    /// Offline forward. `mel`: `(B, n_mels, T)`. Returns `(B, T_out, d_model)`.
    pub fn forward_offline(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.subsample.forward(mel)?;
        let (_b, t_out, _d) = x.dims3()?;
        let pos = rel_position_emb(t_out, self.cfg.d_model, x.device(), x.dtype())?;
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &pos)?;
        }
        Ok(x)
    }
}
