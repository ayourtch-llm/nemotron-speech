# State for the next instance

You are picking up where the previous session left off on this project: a Rust + candle port of NVIDIA's `nemotron-speech-streaming-en-0.6b` (cache-aware FastConformer + RNN-T ASR). The work is in `/Users/ayourtch/rust/nemotron-speech`, branch `main`.

**First thing to do:** `git log --oneline` and you'll see the milestones in order. The codebase is small enough that ~10 minutes of reading puts you back in the seat.

## What works end-to-end (validated)

- File → text via `cargo run --release --bin transcribe -- --audio tmp/small-test.wav --st models/nemotron-speech-streaming-en-0.6b.safetensors --tok models/tokenizer.model --cpu` → "This is a small test to see how the recognition works."
- Same output via `transcribe_streaming` (chunked encoder with KV + conv caches).
- Same output via `transcribe_live` (AudioSource-driven; FileChunkSource works).
- **Mic smoke-tested live** on both CPU and Metal. Clean full-words-only output via the word-initial token detection in `transcribe_live`. Latency feels ~1–2 s per chunk on M1; Metal not noticeably faster than CPU on this workload (small tensors, kernel-launch overhead dominates as previous-me predicted).
- **Offline timing on M1 CPU (5 s clip):** mel 12.5 ms · encoder forward 1.48 s · greedy decode 136 ms. ~3.4× real-time, with ~3× headroom for streaming chunks (~330 ms compute / 1.12 s budget).

## Numerical receipts (don't redo these unless something changes)

| Stage | Max abs vs PyTorch reference | Mean abs |
|---|---|---|
| Mel features (128×500) | 4.7e-5 | 3.5e-7 |
| Subsample output (1, 64, 1024) | 1.95e-3 | 7.9e-5 |
| Layer 0 output | 1.7e-4 | 6.7e-6 |
| Full 24-layer encoder | **1.6e-6** | 5.5e-8 |
| Streaming chunked encoder vs offline+mask | **6.9e-7** | 3.7e-8 |

Validation tools live in `tools/`:
- `convert_nemo.py` — one-shot, `.nemo` → safetensors with key remapping. Already run; outputs are at `models/nemotron-speech-streaming-en-0.6b.safetensors` (2.4 GB, gitignored). **Do not re-run unless the renaming rules change.**
- `reference_mel.py` — Python mel reference. Outputs `tmp/reference_mel.bin` + `.npz`.
- `reference_encoder.py` — PyTorch reimpl of subsample / layer0 / full encoder, reuses the converted safetensors. Has stages: `subsample`, `layer0_ff1`, `layer0_attn`, `layer0`, `encoder`. **No NeMo dependency** — it's a from-scratch port that loads the same weights.
- `mel_check`, `encoder_check`, `streaming_check` (Rust binaries) — Rust diffs against those references.

## Things that would have bitten me had I not figured them out

These are the non-obvious bits — read once and the code makes sense.

1. **`conv.batch_norm` in NeMo's state dict is actually `LayerNorm` for this model**, because `conv_norm_type=layer_norm`. The attribute name `batch_norm` is hardcoded regardless of norm type. No `running_mean`/`running_var` keys → LN, not BN.
2. **The conformer conv module's `pw1`/`dw`/`pw2` have NO bias** (encoder config has `use_bias: false`). The subsampling stack DOES have biases. The two layers that DO have biases on linears anywhere in the encoder side are LayerNorm (always) and the subsampling Conv2d/Linear.
3. **Causal subsampling pads asymmetrically on BOTH freq and time axes** — `(left=k-1, right=s-1) = (2, 1)`. For 128 mel bins, the freq dim shrinks `128 → 65 → 33 → 17` over three stride-2 stages, so the final flatten gives `17×256 = 4352` and the linear is `(1024, 4352)`. The *time* dim also follows this `floor(N/2) + 1` rule.
4. **Position embeddings**: NeMo `RelPositionalEncoding` produces `(1, 2L−1, d)` with positions `[L−1, L−2, …, 0, −1, …, −L+1]`. For cached streaming, `L = T_cache + T_chunk`, not `T_chunk`. The `pos_emb` for chunk N is sized for `2*(cache+chunk)−1`.
5. **`rel_shift` generalizes** to `qlen ≤ klen`. The trick of "pad one zero col, reshape, drop first row, reshape back" works as-is when `qlen < klen`; just slice the last dim to `klen` instead of `qlen`. I verified by hand on (qlen=2, klen=3); see comments in `src/model/encoder.rs::rel_shift`.
6. **NeMo LSTM key naming**: `weight_ih_l0`, `bias_hh_l1`, etc. **Keep the `_l<i>` suffix** when remapping — candle's `lstm()` reads those names directly under its prefix. `tools/convert_nemo.py` already does this.
7. **Chunked-limited attention is chunk-aligned**, not "j ≤ i+R". A frame in chunk `c_i` can attend a frame in chunk `c_j` iff `c_j ∈ [c_i − left_chunks, c_i]`. The `j ≤ i+R` phrasing is approximately right for the first frame of a chunk but loose for later frames. Use chunk IDs.
8. **For our 5 s test clip, the chunked-limited mask is a no-op** — 64 encoded frames / chunk_size 14 = 5 chunks, all within `left_chunks=5` reach. So `--chunked-mask` produces byte-identical output. Need a longer clip to actually test the mask in anger.
9. **Subsampling is causal in time**, so when the front-end re-runs on a growing audio buffer (current `StreamingPipeline` impl), old encoded frames stay stable. The *very last* encoded frame of the buffer depends on right-edge zero padding (s−1 = 1 zero), so it's "tentative" and may shift slightly when audio is extended. For mic streaming this means the boundary frame between chunks may be marginally inaccurate.
10. **The `.nemo` archive** is just a tarball: `model_config.yaml` + tokenizer files + `model_weights.ckpt` (zipped torch state dict). Already extracted to `models/extracted/`. Don't re-extract.
11. **`/tmp/nemo_research/`** has copies of the relevant NeMo source files (audio_preprocessing, conformer_encoder, conformer_modules, multi_head_attention, causal_convs, subsampling, rnnt, rnnt_greedy_decoding, etc.). Faster to read than navigating the live NeMo repo. *Note: `/tmp` may not survive reboots.*

## What's not done

In rough priority order:

1. **Incremental front-end.** Today `StreamingPipeline::advance_chunk` re-runs mel + subsample on the full accumulated audio buffer each chunk. Correct (causal) but O(N) per chunk where N grows with utterance length. Proper streaming wants:
   - `MelExtractor` with state: 1 sample of preemph history, last `n_fft/2` samples for reflection padding (note: right-side reflection at the end of a stream needs *future* samples, so output lags by `n_fft/(2*hop) = 1` mel frame).
   - `DwStridingSubsampling` with per-stage conv state caches: `(k−1)` input frames at each of 3 stages. After 3 strides of 2 with the asymmetric `(2, 1)` pad, the audio-side context needed is small (~8 mel frames ≈ 80 ms).
2. **Longer-utterance test on the CUDA box.** User has a ~4.5 (minutes? something) audio file they'll plug in tomorrow. This is where the chunked-limited mask actually matters and where Metal/CUDA throughput should win over CPU.
3. **UDP audio source.** User mentioned future use case: real-time UDP packet ingestion. The `AudioSource` trait abstraction is set up exactly for this. UDP receiver thread → mpsc → `MicSource`-style consumer.
4. **Performance.** Metal is currently *slower* than CPU on the 5 s clip due to per-op kernel launch overhead on small tensors (T=64). Should win on longer audio or batching. Haven't profiled CUDA at all.
5. **Punctuation smoothing (low priority).** Model emits sentence-final periods at chunk boundaries even mid-thought. Cosmetic — could be improved by holding back trailing punctuation tokens too, but diminishing returns.

## User context (from this session)

- User: Andrew Yourtchenko, ayourtch@gmail.com.
- Style: enthusiastic, lightweight tone, asks "what do you think?" exploratory questions and likes options + tradeoffs.
- They like commits at testable milestones — I committed at every "this works and is validated" moment (8 commits). Continue this pattern.
- They have **codex** available in **pty-2** as a second-opinion model. I never needed to consult it but the offer stands.
- They asked for ALL THREE devices eventually (CPU/Metal/CUDA); Metal is the immediate target on their Mac. CUDA will be tested on a separate box.
- They mentioned UDP audio packets as the eventual real-time use case.
- They're playful in chat — match the tone, but in code/commits stay matter-of-fact.

## Repo layout (what's where)

```
.
├── Cargo.toml                  default features = ["cpu"]; "metal", "cuda", "mic" optional
├── tools/
│   ├── convert_nemo.py         .nemo .ckpt -> safetensors (DONE; output already on disk)
│   ├── reference_mel.py        Python mel reference (writes tmp/reference_mel.bin)
│   └── reference_encoder.py    Python encoder reference (writes tmp/ref_*.bin)
├── models/                     gitignored; contains .nemo, .safetensors, tokenizer.model, extracted/
├── tmp/                        gitignored; small-test.wav (5s) + reference dumps
├── src/
│   ├── main.rs                 stub; the real binaries are in src/bin/
│   ├── lib.rs                  module roots
│   ├── audio.rs                symphonia loader (wav/m4a -> 16 kHz mono f32)
│   ├── audio_source.rs         AudioSource trait + FileChunkSource + MicSource (mic feature)
│   ├── features.rs             log-mel matching the model's preprocessor
│   ├── streaming.rs            StreamingPipeline: push_audio + try_advance
│   ├── tokenizer.rs            SentencePiece detokenize wrapper
│   ├── model/
│   │   ├── mod.rs              ModelConfig (hardcoded for this checkpoint), RuntimeConfig
│   │   ├── encoder.rs          DwStridingSubsampling, ConformerLayer, RelPosMha,
│   │   │                       ConvModule, FastConformerEncoder, EncoderCache
│   │   ├── predict.rs          2-layer LSTM prediction net
│   │   ├── joint.rs            Split-projection joint network
│   │   └── greedy.rs           Greedy RNN-T decoding (single-stream)
│   └── bin/
│       ├── mel_check.rs        Rust mel diff vs Python reference
│       ├── encoder_check.rs    Rust encoder diff vs Python reference (subsample/layer0/encoder)
│       ├── streaming_check.rs  Streaming-encoder diff vs offline+mask
│       ├── transcribe.rs       Original offline file -> text
│       ├── transcribe_streaming.rs  Same but encoder runs chunked with caches
│       └── transcribe_live.rs       AudioSource-driven (file or mic)
```

## Streaming output formatting (transcribe_live)

`transcribe_live` emits one `[chunk] <text>` line per chunk that has new content. To avoid mid-word splits at chunk boundaries (`[chunk] t` / `[chunk] erminal`), it holds back the trailing partial-word run and only emits text up to the most recent word-initial SentencePiece token. Word-initial detection is via detokenize-diff: a token is word-initial iff appending it adds a leading space to the running text. End-of-stream (`is_final`) flushes any held-back tail. See `last_word_initial()` in the binary.

Trade-off: the last word of an utterance only appears once the *next* word starts (or on stream end). Acceptable for live dictation; visible in logs as a steady one-word-trailing display.

## Last commit

`git log -1 --oneline` should show `transcribe_live: hold back partial words at chunk boundaries` (or similar). The tree should be clean. If it's not, look at `git status` first — the user might have started something.

## Things to NOT do

- Don't re-download the 2.4 GB `.nemo`. It's at `models/nemotron-speech-streaming-en-0.6b.nemo` and gitignored.
- Don't re-extract the `.nemo`. It's at `models/extracted/`.
- Don't rerun the converter unless the renaming rules change. The safetensors is at `models/nemotron-speech-streaming-en-0.6b.safetensors`.
- Don't try to install NeMo. The Python reference scripts are deliberately NeMo-free — they reuse the converted safetensors and reimplement the math from the source files in `/tmp/nemo_research/`.
- Don't write a doc file unless the user asks for it (this `state.md` and `README.md` are the only ones, both explicitly requested).
- Don't run the mic binary autonomously — it grabs the system mic and you have no way to feed it audio.

Good luck. Ask if anything's unclear; the user is pretty engaged and will course-correct.
