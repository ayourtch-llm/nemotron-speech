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

    /// Streaming variant: feeds the next slice of mel frames through the
    /// stack while keeping a per-stage rolling buffer of unconsumed
    /// pre-conv input frames. Initial buffer = 2 zero frames per stage
    /// (matching offline's `pad_causal2d` left-zero-pad of size 2).
    ///
    /// Output is appended to `state.output`. Only "stable" frames whose
    /// right-context is real are emitted mid-stream. When `finished ==
    /// true`, a right-zero-pad of 1 is applied at each stage so the
    /// cumulative output exactly matches `forward()` on the same total
    /// mel input.
    pub fn forward_incremental(
        &self,
        new_mel: &Tensor, // (1, n_mels, T_new)
        state: &mut SubsampleStreamingState,
        device: &Device,
        dtype: DType,
        finished: bool,
    ) -> Result<()> {
        if state.finalized {
            return Ok(());
        }
        let (b, m, _t_new) = new_mel.dims3()?;
        debug_assert_eq!(m, self.n_mels);
        let mut x = new_mel.transpose(1, 2)?.unsqueeze(1)?; // (1, 1, T_new, n_mels)

        // Stage 0 — input channel = 1, freq dim = n_mels
        if state.stage0_buf.is_none() {
            state.stage0_buf = Some(Tensor::zeros((b, 1, 2, self.n_mels), dtype, device)?);
        }
        x = self.stage_run(
            x,
            state.stage0_buf.as_mut().unwrap(),
            &self.conv0,
            None,
            true,
            finished,
        )?;

        // Stage 1
        let c = x.dim(1)?;
        let f = x.dim(3)?;
        if state.stage1_buf.is_none() {
            state.stage1_buf = Some(Tensor::zeros((b, c, 2, f), dtype, device)?);
        }
        x = self.stage_run(
            x,
            state.stage1_buf.as_mut().unwrap(),
            &self.dw1,
            Some(&self.pw1),
            true,
            finished,
        )?;

        // Stage 2
        let c = x.dim(1)?;
        let f = x.dim(3)?;
        if state.stage2_buf.is_none() {
            state.stage2_buf = Some(Tensor::zeros((b, c, 2, f), dtype, device)?);
        }
        x = self.stage_run(
            x,
            state.stage2_buf.as_mut().unwrap(),
            &self.dw2,
            Some(&self.pw2),
            true,
            finished,
        )?;

        // (B, C, T_new_2, F') -> (B, T_new_2, C*F') -> linear
        let (_b, c2, tp, f2) = x.dims4()?;
        if tp == 0 {
            state.finalized |= finished;
            return Ok(());
        }
        let new_enc = x.permute((0, 2, 1, 3))?.contiguous()?.reshape((b, tp, c2 * f2))?;
        let new_enc = self.out.forward(&new_enc)?; // (1, T_new_2, d_model)

        let new_emit = new_enc.dim(1)?;
        state.n_emitted += new_emit;
        state.output = match state.output.take() {
            Some(prev) => Some(Tensor::cat(&[prev, new_enc], 1)?),
            None => Some(new_enc),
        };
        state.finalized |= finished;
        Ok(())
    }

    /// Run one subsample stage incrementally on a rolling buffer.
    ///
    /// `buf` is the unconsumed tail of this stage's input sequence (along
    /// the time axis), pre-freq-pad. Initially it holds 2 zero frames (=
    /// offline's left-pad-2). On entry we append `x` to it, optionally
    /// append a single zero on the right when `finished`, run the conv
    /// (stride 2, k=3) — which produces `(t_buf - 3)/2 + 1` stable output
    /// frames — and drop `2 * n_emitted` input positions from the front
    /// of `buf` so the next call's first read aligns with where the conv
    /// left off.
    fn stage_run(
        &self,
        x: Tensor,                    // (B, C_in, T_new, F_in)
        buf: &mut Tensor,             // (B, C_in, T_buf, F_in)
        dw_or_conv0: &Conv2d,
        pw: Option<&Conv2d>,
        relu_after: bool,
        finished: bool,
    ) -> Result<Tensor> {
        // 1) Append new frames.
        let extended = Tensor::cat(&[&*buf, &x], 2)?;

        // 2) Optionally append right-pad-1 (only when finished).
        let to_conv = if finished {
            extended.pad_with_zeros(D::Minus2, 0, 1)?
        } else {
            extended.clone()
        };
        let t_conv = to_conv.dim(2)?;

        // 3) Number of stable output frames.
        let n_emit = if t_conv >= 3 { (t_conv - 3) / 2 + 1 } else { 0 };
        if n_emit == 0 {
            // Update buf to include the new frames; nothing emitted this call.
            *buf = extended;
            // Empty stage output: shape (B, C_out, 0, F_out). Build a zero
            // tensor of the right shape via the existing freq-padded shape,
            // run conv on a 3-frame slice. But there's none — return the
            // properly-shaped zero. Since the caller must accept any T_new,
            // and downstream stages will short-circuit at tp==0 too, return
            // the simplest tensor with T=0.
            // Easiest: construct via narrow-with-len-0 from a forward call
            // on a minimum-length input — but that's surgery. Cheat by
            // running conv on `to_conv` if t_conv >= 3, otherwise produce
            // shape via a zero-frame output. We branch on n_emit, so
            // t_conv < 3 here. Build a (B, C_in, 0, F_freq_pad) zero and
            // forward the conv to discover output shape.
            let device = to_conv.device();
            let dtype = to_conv.dtype();
            let f_in_padded = to_conv.dim(3)? + 3;
            let probe = Tensor::zeros(
                (to_conv.dim(0)?, to_conv.dim(1)?, 3, f_in_padded),
                dtype,
                device,
            )?;
            let probe_out = dw_or_conv0.forward(&probe)?;
            let probe_out = if let Some(pw) = pw { pw.forward(&probe_out)? } else { probe_out };
            let f_out = probe_out.dim(3)?;
            let c_out = probe_out.dim(1)?;
            return Ok(Tensor::zeros((to_conv.dim(0)?, c_out, 0, f_out), dtype, device)?);
        }

        // 4) Update buf for next call (skip if finished — no further calls).
        if !finished {
            let t_ext = extended.dim(2)?;
            let drop = 2 * n_emit;
            let keep = t_ext - drop;
            *buf = extended.narrow(2, drop, keep)?.contiguous()?;
        }

        // 5) Convolve.
        let to_conv = to_conv.pad_with_zeros(D::Minus1, 2, 1)?;
        let mut y = dw_or_conv0.forward(&to_conv)?;
        if let Some(pw) = pw {
            y = pw.forward(&y)?;
        }
        if relu_after {
            y = y.relu()?;
        }
        Ok(y)
    }
}

/// Rolling per-stage input buffers + accumulating output for the streaming subsample.
pub struct SubsampleStreamingState {
    stage0_buf: Option<Tensor>,
    stage1_buf: Option<Tensor>,
    stage2_buf: Option<Tensor>,
    pub output: Option<Tensor>, // (1, T_total_emitted, d_model)
    pub n_emitted: usize,
    finalized: bool,
}

impl SubsampleStreamingState {
    pub fn empty() -> Self {
        Self {
            stage0_buf: None,
            stage1_buf: None,
            stage2_buf: None,
            output: None,
            n_emitted: 0,
            finalized: false,
        }
    }

    pub fn is_finalized(&self) -> bool {
        self.finalized
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
        let (out, _) = self.forward_with_cache(x, None)?;
        Ok(out)
    }

    pub fn kernel(&self) -> usize {
        self.kernel
    }
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Cache-aware forward. `cache_in` is the post-GLU activations from the
    /// last `kernel-1` time steps of the previous chunk (`(B, d, kernel-1)`).
    /// Returns the layer output and the next cache (sized `kernel-1`).
    /// When `cache_in` is `None`, behaves exactly like the offline path
    /// (zero-pads the left edge by `kernel-1`).
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        cache_in: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, t, d) = x.dims3()?;
        debug_assert_eq!(d, self.d_model);
        // (B, T, d) -> (B, d, 1, T)
        let x = x.transpose(1, 2)?.unsqueeze(2)?;
        // pointwise -> (B, 2d, 1, T)
        let x = self.pw1.forward(&x)?;
        // GLU along channel dim: split (a, b) on channel, gated = a * sigmoid(b)
        let half = self.d_model;
        let a = x.narrow(1, 0, half)?;
        let bg = x.narrow(1, half, half)?;
        let glu = a.broadcast_mul(&candle_nn::ops::sigmoid(&bg)?)?; // (B, d, 1, T)

        // depthwise input: prepend cache (or zero pad) on time
        let glu_2d = glu.squeeze(2)?; // (B, d, T)
        let next_cache = {
            // The next conv cache is the last (kernel-1) frames of the post-GLU
            // activations for THIS chunk, padded with the existing cache on
            // the left if the chunk is shorter than (kernel-1).
            let need = self.kernel - 1;
            if t >= need {
                glu_2d.narrow(D::Minus1, t - need, need)?.contiguous()?
            } else {
                // Concatenate (cache_in_or_zeros[t..]) with glu_2d.
                let cache_existing = match cache_in {
                    Some(c) => c.clone(),
                    None => Tensor::zeros((b, d, need), x.dtype(), x.device())?,
                };
                Tensor::cat(&[
                    &cache_existing.narrow(D::Minus1, t, need - t)?,
                    &glu_2d,
                ], D::Minus1)?.contiguous()?
            }
        };

        let prefix = match cache_in {
            Some(c) => c.clone(),
            None => Tensor::zeros((b, d, self.kernel - 1), x.dtype(), x.device())?,
        };
        let padded = Tensor::cat(&[&prefix, &glu_2d], D::Minus1)?.contiguous()?; // (B, d, k-1 + T)
        // back to 4D for Conv2d
        let padded4 = padded.unsqueeze(2)?;
        let x = self.dw.forward(&padded4)?;
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
        Ok((xn, next_cache))
    }
}

// ---------------------------------------------------------------------------
// Chunked-limited attention mask (additive form: 0 for visible, very
// negative for masked).
// ---------------------------------------------------------------------------

/// Build an additive `(T, T)` attention mask shaped to broadcast over
/// `(B, h, T, T)` scores. `chunk_size = att_context_size[1] + 1`,
/// `left_chunks = att_context_size[0] / chunk_size`.
///
/// `mask[i][j] = 0`  if frame j is visible from frame i
/// `mask[i][j] = -1e9` otherwise
///
/// Visibility rule (chunk-aligned):  c_j ∈ [c_i - left_chunks, c_i].
pub fn chunked_limited_mask(
    t: usize,
    chunk_size: usize,
    left_chunks: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut data = vec![0.0f32; t * t];
    let neg_inf = -1e9f32;
    for i in 0..t {
        let c_i = i / chunk_size;
        for j in 0..t {
            let c_j = j / chunk_size;
            let visible = c_j <= c_i && c_i - c_j <= left_chunks;
            if !visible {
                data[i * t + j] = neg_inf;
            }
        }
    }
    let m = Tensor::from_vec(data, (1, 1, t, t), device)?.to_dtype(dtype)?;
    Ok(m)
}

// ---------------------------------------------------------------------------
// Multi-Head Attention with relative positional embeddings (Transformer-XL)
// ---------------------------------------------------------------------------

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
    /// `attn_bias`: optional additive mask, broadcastable to `(B, h, T, T)`.
    pub fn forward(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        attn_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
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
        let matrix_bd = rel_shift(&matrix_bd_raw, t, t)?; // (B, h, T, T)

        let scale = 1.0 / (dh as f64).sqrt();
        let scores = (matrix_ac + matrix_bd)?.affine(scale, 0.0)?;
        let scores = match attn_bias {
            Some(bias) => scores.broadcast_add(bias)?,
            None => scores,
        };

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v)?; // (B, h, T, dh)
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, h * dh))?;
        let out = self.out.forward(&ctx)?;
        Ok(out)
    }

    /// Cache-aware forward.
    ///
    /// `x` is the current chunk `(B, T_q, d_model)`.
    /// `kv_cache_in` is the previous K/V context, shape `(B, T_cache, d_model)`,
    /// or `None` for the first chunk. The returned `kv_full` is `(cache, x)`
    /// concatenated; the caller is responsible for trimming it to its policy
    /// (e.g. keep last `att_context_size[0]` frames).
    ///
    /// `pos_emb` must be sized for `klen = T_cache + T_q` total positions
    /// (i.e. shape `(1, 2*klen - 1, d_model)`).
    pub fn forward_chunked(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        kv_cache_in: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, t_q, _d) = x.dims3()?;
        let h = self.n_heads;
        let dh = self.d_head;

        let kv_in = match kv_cache_in {
            Some(c) => Tensor::cat(&[c, x], 1)?.contiguous()?,
            None => x.clone(),
        };
        let t_kv = kv_in.dims3()?.1;

        let q = self
            .q
            .forward(x)?
            .reshape((b, t_q, h, dh))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(&kv_in)?
            .reshape((b, t_kv, h, dh))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(&kv_in)?
            .reshape((b, t_kv, h, dh))?
            .transpose(1, 2)?
            .contiguous()?;

        let pos_len = pos_emb.dims3()?.1;
        debug_assert_eq!(pos_len, 2 * t_kv - 1);
        let p = self
            .pos
            .forward(pos_emb)?
            .reshape((1, pos_len, h, dh))?
            .transpose(1, 2)?
            .contiguous()?;

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

        // ac: (B, h, T_q, T_kv)
        let matrix_ac = q_u.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        // bd_raw: (B, h, T_q, 2*T_kv - 1) -> rel_shift -> (B, h, T_q, T_kv)
        let p_kt = p.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let matrix_bd_raw = q_v.broadcast_matmul(&p_kt)?;
        let matrix_bd = rel_shift(&matrix_bd_raw, t_q, t_kv)?;

        let scale = 1.0 / (dh as f64).sqrt();
        let scores = (matrix_ac + matrix_bd)?.affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t_q, h * dh))?;
        let out = self.out.forward(&ctx)?;
        Ok((out, kv_in))
    }
}

/// Transformer-XL relative shift, generalized to arbitrary `qlen <= klen`.
/// Input shape: `(B, h, qlen, 2*klen-1)`. Output: `(B, h, qlen, klen)`.
///
/// The square case (qlen == klen) is what the offline path uses; the
/// non-square case (qlen < klen) is what cache-aware streaming uses, where
/// `klen = qlen + cache_len`.
fn rel_shift(x: &Tensor, qlen: usize, klen: usize) -> Result<Tensor> {
    let (b, h, q1, l) = x.dims4()?;
    debug_assert_eq!(q1, qlen);
    debug_assert_eq!(l, 2 * klen - 1);
    let zero = Tensor::zeros((b, h, qlen, 1), x.dtype(), x.device())?;
    let x = Tensor::cat(&[&zero, x], D::Minus1)?; // (B, h, qlen, 2*klen)
    let x = x.contiguous()?.reshape((b, h, 2 * klen, qlen))?;
    let x = x.narrow(D::Minus2, 1, 2 * klen - 1)?; // (B, h, 2*klen-1, qlen)
    let x = x.contiguous()?.reshape((b, h, qlen, 2 * klen - 1))?;
    let x = x.narrow(D::Minus1, 0, klen)?;
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

    pub fn forward(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        attn_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        // FF1 macaron half
        let r = x.clone();
        let y = self.norm_ff1.forward(x)?;
        let y = self.ff1.forward(&y)?.affine(0.5, 0.0)?;
        let x = (r + y)?;

        // Self-attention
        let r = x.clone();
        let y = self.norm_attn.forward(&x)?;
        let y = self.attn.forward(&y, pos_emb, attn_bias)?;
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

    /// Cache-aware forward. `cache` is read AND written: kv updated to the
    /// concatenation of `(old_kv, x)` (caller trims), conv updated to the
    /// last (kernel-1) frames of post-GLU activations.
    pub fn forward_chunked(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        cache: &mut LayerCache,
    ) -> Result<Tensor> {
        // FF1 macaron half
        let r = x.clone();
        let y = self.norm_ff1.forward(x)?;
        let y = self.ff1.forward(&y)?.affine(0.5, 0.0)?;
        let x = (r + y)?;

        // Self-attention with KV cache
        let r = x.clone();
        let y = self.norm_attn.forward(&x)?;
        let (y, kv_full) = self.attn.forward_chunked(&y, pos_emb, cache.kv.as_ref())?;
        cache.kv = Some(kv_full);
        let x = (r + y)?;

        // Conv with state cache
        let r = x.clone();
        let y = self.norm_conv.forward(&x)?;
        let (y, conv_next) = self.conv.forward_with_cache(&y, cache.conv.as_ref())?;
        cache.conv = Some(conv_next);
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
// Per-layer streaming cache.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct LayerCache {
    /// (B, T_kv, d_model). `None` at start; grows up to `max_kv` then slides.
    pub kv: Option<Tensor>,
    /// (B, d_model, kernel-1). Post-pw1+GLU activations from previous chunk.
    pub conv: Option<Tensor>,
}

#[derive(Clone, Debug, Default)]
pub struct EncoderCache {
    pub layers: Vec<LayerCache>,
}

impl EncoderCache {
    pub fn empty(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| LayerCache::default()).collect(),
        }
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

    /// Offline forward with full attention. `mel`: `(B, n_mels, T)`.
    pub fn forward_offline(&self, mel: &Tensor) -> Result<Tensor> {
        self.forward_full(mel, /* with_chunked_mask = */ false)
    }

    /// Offline forward over the whole utterance, optionally applying the
    /// trained chunked-limited attention mask. Functionally identical to the
    /// streaming pass when caches are unbounded; used as the validation
    /// reference for the chunk-by-chunk path.
    /// Run the encoder layers on a chunk of pre-subsampled features.
    /// `encoded_chunk`: `(B, T_chunk, d_model)`.
    /// `cache`: per-layer caches; updated in place.
    /// Returns: encoder output for this chunk `(B, T_chunk, d_model)`.
    pub fn forward_layers_chunked(
        &self,
        encoded_chunk: &Tensor,
        cache: &mut EncoderCache,
    ) -> Result<Tensor> {
        let (_b, t_q, _d) = encoded_chunk.dims3()?;
        // klen for this chunk is current cache length (from layer 0; all layers
        // share kv cache length) plus this chunk's length.
        let cache_len = cache
            .layers
            .first()
            .and_then(|l| l.kv.as_ref())
            .map(|t| t.dims3().map(|(_, n, _)| n).unwrap_or(0))
            .unwrap_or(0);
        let klen = cache_len + t_q;
        let pos = rel_position_emb(klen, self.cfg.d_model, encoded_chunk.device(), encoded_chunk.dtype())?;

        let max_kv = self.cfg.att_context_size[0];
        let mut x = encoded_chunk.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_chunked(&x, &pos, &mut cache.layers[i])?;
            // Trim KV cache: keep last `max_kv` frames.
            if let Some(kv) = cache.layers[i].kv.take() {
                let (_, n, _) = kv.dims3()?;
                let trimmed = if n > max_kv {
                    kv.narrow(1, n - max_kv, max_kv)?.contiguous()?
                } else {
                    kv
                };
                cache.layers[i].kv = Some(trimmed);
            }
        }
        Ok(x)
    }

    pub fn forward_full(&self, mel: &Tensor, with_chunked_mask: bool) -> Result<Tensor> {
        let x = self.subsample.forward(mel)?;
        let (_b, t_out, _d) = x.dims3()?;
        let pos = rel_position_emb(t_out, self.cfg.d_model, x.device(), x.dtype())?;
        let mask = if with_chunked_mask {
            Some(chunked_limited_mask(
                t_out,
                self.cfg.chunk_size_enc_frames(),
                self.cfg.left_chunks(),
                x.device(),
                x.dtype(),
            )?)
        } else {
            None
        };
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &pos, mask.as_ref())?;
        }
        Ok(x)
    }
}
