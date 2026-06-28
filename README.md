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
# Download + extract + convert in one go (idempotent; safe to re-run).
bash tools/get_model.sh

# Then transcribe a wav file:
cargo run --release --bin transcribe -- \
    --audio path/to/some.wav \
    --st  models/nemotron-speech-streaming-en-0.6b.safetensors \
    --tok models/tokenizer.model --cpu
```

Or run the steps manually:

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
```

## Binaries

| Binary | What it does |
|---|---|
| `transcribe` | File → text via the offline encoder. The simplest path. |
| `transcribe_streaming` | File → text but the encoder runs chunk-by-chunk with KV + conv caches. Same output as `transcribe`. |
| `transcribe_live` | Audio-source-driven streaming. Files (`--audio`), mic (`--features mic --mic`), or UDP receiver (`--udp-listen <addr>`). |
| `udp_mic_send` | Companion sender (with `--features mic`): captures mic, sends raw f32-LE 16 kHz mono PCM to a `--target host:port`. Pairs with `transcribe_live --udp-listen`. |
| `mel_check` | Diff Rust mel vs Python mel reference. |
| `encoder_check` | Diff Rust encoder vs Python encoder reference, by stage (`subsample`, `layer0`, `encoder`). |
| `streaming_check` | Diff streaming encoder output vs offline+mask reference. |
| `subsample_check` | Diff streaming subsample vs offline `forward()` across slice sizes. |

## Build features

```toml
default    = ["cpu"]
accelerate = ["candle-core/accelerate", "candle-nn/accelerate"]  # Apple BLAS (macOS CPU)
metal      = ["candle-core/metal", "candle-nn/metal"]
cuda       = ["candle-core/cuda",  "candle-nn/cuda"]
mic        = [...]   # cpal microphone source
```

```sh
cargo run --release --no-default-features --features accelerate --bin transcribe -- ...
cargo run --release --features metal --bin transcribe -- ...
cargo run --release --features cuda  --bin transcribe -- ...
cargo run --release --features mic   --bin transcribe_live -- --mic ...
```

Device selection at runtime: `--cpu` forces CPU even when a GPU feature is enabled; otherwise the binary tries Metal/CUDA first and falls back to CPU.

On macOS CPU, build with `--no-default-features --features accelerate` to route matmuls through Apple's Accelerate BLAS (~1.6× on the matmul-bound offline path).

## Chunk batching (`--chunk-batch`, throughput vs latency)

`transcribe_live` accepts `--chunk-batch N` (default 1): it fuses **N encoder chunks into one pass** (both the subsample stack and the conformer layers), amortising per-op dispatch overhead. A block-causal attention mask makes the result **byte-equivalent** to processing one chunk at a time — verified by `streaming_check --batch N` (max abs diff vs the offline+mask reference stays ~4e-6 for N up to 8).

This is what makes **Metal usable for live audio**. Per-chunk (N=1), Metal launches thousands of tiny GPU ops/sec and runs *slower than real time* — the original ~80%-audio-drop failure mode. Batching turns those into a few large dispatches:

| `--chunk-batch` | Metal, 40 s clip (incl. load) | realtime | worst-case latency |
|---|---|---|---|
| 1 | ~45 s | ~1.0× (drops audio) | ~1.1 s |
| 2 | ~26 s | ~1.85× | ~2.2 s |
| 8 | ~13.5 s | ~4.2× | ~9 s |
| 16 | ~7.5 s | ~5.3× | ~18 s |

The trade is **linear: latency ≈ N × 1.12 s** worst-case, because the pipeline waits for N whole chunks (1.12 s each) before emitting a burst. So high N is for file/batch transcription or catching up; for a conversational loop use **N=2** (keeps up on Metal at ~1.1 s average latency, leaving the CPU nearly free). On CPU every N keeps up — there, N>1 only matters to free cores for other work.

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
├── audio_source.rs    AudioSource trait + FileChunkSource + MicSource + UdpSource
├── features.rs        log-mel (offline + IncrementalMelExtractor with preemph state)
├── streaming.rs       StreamingPipeline (push_audio + try_advance, fully incremental)
├── tokenizer.rs       SentencePiece wrapper
├── model/
│   ├── mod.rs           ModelConfig
│   ├── encoder.rs       DwStridingSubsampling (offline + streaming with rolling
│   │                    per-stage buffers), ConformerLayer, RelPosMha, ConvModule,
│   │                    FastConformerEncoder, EncoderCache, SubsampleStreamingState
│   ├── predict.rs       2-layer LSTM prediction net (candle_nn::lstm)
│   ├── joint.rs         Split-projection joint network
│   └── greedy.rs        Greedy RNN-T decoding (single-stream, blank-as-pad)
└── bin/                 (see "Binaries" above)

tools/
├── get_model.sh         Idempotent: download + extract + convert + tokenizer
├── convert_nemo.py      One-shot .nemo -> safetensors with key remapping
├── reference_mel.py     PyTorch mel reference; writes a flat .bin Rust can diff against
└── reference_encoder.py PyTorch reimpl of subsample / layer0 / full encoder using saved weights
```

## What's still on the list

- **Mic source: stop dropping samples.** `MicSource::open_default` uses `tx.try_send` which silently drops when the channel backs up. Switching to `blocking_send` or growing the depth-64 channel would prevent live-mic drops. (The other half of the old Metal failure — per-op overhead — is now addressed by `--chunk-batch`; this part is independent and still open.)
- **Adaptive `--chunk-batch`.** Today N is fixed. Batching N=1 at the live edge and raising N only when a backlog builds would give high-N resilience at N=1 latency in steady state.
- **Longer-utterance test.** Done for benchmarking via a repeated-clip 40 s file (exercises the chunked-limited attention mask, which is a no-op on the 5 s clip). A real multi-minute clip would let CUDA pull ahead of CPU on raw throughput.

## License

The model weights are NVIDIA's, governed by their terms — see the [Hugging Face model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b). The Rust source in this repository is licensed under [Apache-2.0](LICENSE).
