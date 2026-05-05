# nemotron-speech

A Rust + [candle](https://github.com/huggingface/candle) port of NVIDIA's [`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) — a cache-aware FastConformer + RNN-T streaming ASR model. Targets CPU, Metal, and CUDA from one codebase.

```
$ cargo run --release --bin transcribe -- \
    --audio tmp/small-test.wav \
    --st  models/nemotron-speech-streaming-en-0.6b.safetensors \
    --tok models/tokenizer.model --cpu
text: This is a small test to see how the recognition works.
```

## Status

Validated end-to-end. The chunked-streaming encoder produces output bit-equivalent to the offline-with-mask reference (max abs diff `6.9e-7` over `(1, 64, 1024)` post-LayerNorm activations).

| Stage | Validated against | Max abs diff |
|---|---|---|
| Log-mel features | PyTorch reimpl using saved Hann window + mel filterbank | 4.7e-5 |
| Full 24-layer encoder (offline) | PyTorch reimpl from saved weights | 1.6e-6 |
| Streaming chunked encoder | offline-with-chunked-mask | 6.9e-7 |

Each row's reference is reproducible by running the corresponding script in [`tools/`](tools/) and the matching Rust binary in [`src/bin/`](src/bin/).

## Quick start

You'll need the model from Hugging Face (~2.4 GB) and a Python with `torch`, `safetensors`, and `numpy` available for the one-shot weight conversion.

```sh
# 1. Download the model (gitignored under models/)
mkdir -p models
curl -L -o models/nemotron-speech-streaming-en-0.6b.nemo \
    https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/resolve/main/nemotron-speech-streaming-en-0.6b.nemo

# 2. Extract (.nemo is just a tarball)
mkdir -p models/extracted
tar -xf models/nemotron-speech-streaming-en-0.6b.nemo -C models/extracted

# 3. Convert weights to safetensors and copy out the SentencePiece model
python3 tools/convert_nemo.py \
    --ckpt models/extracted/model_weights.ckpt \
    --out  models/nemotron-speech-streaming-en-0.6b.safetensors
cp models/extracted/*tokenizer.model models/tokenizer.model

# 4. Transcribe
cargo run --release --bin transcribe -- \
    --audio tmp/small-test.wav \
    --st  models/nemotron-speech-streaming-en-0.6b.safetensors \
    --tok models/tokenizer.model --cpu
```

## Binaries

| Binary | What it does |
|---|---|
| `transcribe` | File → text via the offline encoder. The simplest path. |
| `transcribe_streaming` | File → text but the encoder runs chunk-by-chunk with KV + conv caches. Same output as `transcribe`. |
| `transcribe_live` | Audio-source-driven streaming. Works for files (default) and mic (`--features mic --mic`). UDP source slot is set up but not implemented. |
| `mel_check` | Diff Rust mel vs Python mel reference. |
| `encoder_check` | Diff Rust encoder vs Python encoder reference, by stage (`subsample`, `layer0`, `encoder`). |
| `streaming_check` | Diff streaming encoder output vs offline+mask reference. |

## Build features

```toml
default = ["cpu"]
metal   = ["candle-core/metal", "candle-nn/metal"]
cuda    = ["candle-core/cuda",  "candle-nn/cuda"]
mic     = [...]   # cpal microphone source
```

```sh
cargo run --release --features metal --bin transcribe -- ...
cargo run --release --features cuda  --bin transcribe -- ...
cargo run --release --features mic   --bin transcribe_live -- --mic ...
```

Device selection at runtime: `--cpu` forces CPU even when a GPU feature is enabled; otherwise the binary tries Metal/CUDA first and falls back to CPU.

## Model architecture (this checkpoint)

| | |
|---|---|
| Encoder | 24-layer cache-aware FastConformer (rel-pos MHA, GLU + causal depthwise conv, macaron) |
| `d_model` / heads / d_head | 1024 / 8 / 128 |
| Subsampling | 8× via three `dw_striding` stages (causal-pad both axes, k=3, s=2) |
| Mel | 128 bins @ 16 kHz, n_fft 512, win 400, hop 160, log + 0.97 preemph |
| Decoder | 2-layer LSTM prediction net (hidden 640) + split-projection joint (640) |
| Vocab | 1024 BPE (SentencePiece) + blank |
| Streaming | `att_context_size = [70, 13]` (default; ~1.12 s lookahead, 70-frame KV cache) |
| Total params | ~600 M |

The streaming math is whatever the trained `cache_aware_rnnt.yaml` recipe specifies: chunk size = R+1 = 14 encoded frames (≈ 1.12 s), left context = 70 frames (= 5 chunks), KV cache size = 70 frames per layer, conv state cache = `kernel - 1 = 8` frames per layer.

## Repo layout

```
src/
├── audio.rs           symphonia file loader -> 16 kHz mono f32
├── audio_source.rs    AudioSource trait + FileChunkSource + MicSource (mic feature)
├── features.rs        log-mel matching the model's preprocessor exactly
├── streaming.rs       StreamingPipeline (push_audio + try_advance)
├── tokenizer.rs       SentencePiece wrapper
├── model/
│   ├── mod.rs           ModelConfig
│   ├── encoder.rs       DwStridingSubsampling, ConformerLayer, RelPosMha,
│   │                    ConvModule, FastConformerEncoder, EncoderCache
│   ├── predict.rs       2-layer LSTM prediction net (candle_nn::lstm)
│   ├── joint.rs         Split-projection joint network
│   └── greedy.rs        Greedy RNN-T decoding (single-stream, blank-as-pad)
└── bin/                 (see "Binaries" above)

tools/
├── convert_nemo.py      One-shot .nemo -> safetensors with key remapping
├── reference_mel.py     PyTorch mel reference; writes a flat .bin Rust can diff against
└── reference_encoder.py PyTorch reimpl of subsample / layer0 / full encoder using saved weights
```

## What's deliberately not done yet

- **Incremental front-end.** `StreamingPipeline::advance_chunk` currently re-runs mel + subsample over the full accumulated audio buffer each chunk. Correct (subsample is causal) but O(N) per chunk. A streaming front-end (preemph state + reflect-pad state + per-stage 2D conv cache) is the next optimization. The encoder/decoder side is already cache-aware.
- **Mic smoke test.** `transcribe_live --features mic --mic` compiles but isn't part of the validated path yet.
- **UDP audio source.** Trait slot exists; implementation pending.
- **Performance tuning.** Metal currently loses to CPU on the 5 s test clip due to kernel-launch overhead on tiny tensors. Should win on longer utterances. CUDA hasn't been profiled at all.

## License

The model weights are NVIDIA's, governed by their terms — see the [Hugging Face model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b). The Rust source in this repository is © its contributors; pick a license that suits your use.
