#!/usr/bin/env bash
# Download + extract + convert NVIDIA's nemotron-speech-streaming-en-0.6b
# checkpoint into the safetensors layout this repo expects.
#
# Idempotent: each step skips if its output already exists. Re-run safely
# at any point.
#
# Outputs (in models/):
#   nemotron-speech-streaming-en-0.6b.nemo            (~2.4 GB tarball, gitignored)
#   extracted/                                         (unpacked .nemo contents)
#   nemotron-speech-streaming-en-0.6b.safetensors     (~2.3 GB, gitignored)
#   tokenizer.model                                    (SentencePiece model, gitignored)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NEMO_URL="https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/resolve/main/nemotron-speech-streaming-en-0.6b.nemo"
NEMO_PATH="models/nemotron-speech-streaming-en-0.6b.nemo"
EXTRACT_DIR="models/extracted"
SAFETENSORS="models/nemotron-speech-streaming-en-0.6b.safetensors"
TOKENIZER="models/tokenizer.model"

step() { printf '\n\033[1;34m==>\033[0m %s\n' "$1"; }
ok()   { printf '    \033[1;32m✓\033[0m %s\n' "$1"; }
note() { printf '    %s\n' "$1"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$1" >&2; exit 1; }

# 0) Tooling check
step "Checking tooling"
command -v curl    >/dev/null 2>&1 || die "curl not found"
command -v tar     >/dev/null 2>&1 || die "tar not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"
python3 -c 'import torch, safetensors' 2>/dev/null \
    || die "Python missing torch and/or safetensors. Install with: python3 -m pip install torch safetensors"
ok "curl, tar, python3, torch, safetensors"

# 1) Download .nemo
mkdir -p models
if [[ -f "$NEMO_PATH" ]]; then
    size=$(stat -c%s "$NEMO_PATH" 2>/dev/null || stat -f%z "$NEMO_PATH")
    if (( size > 2000000000 )); then
        step "Skipping download (.nemo already at $NEMO_PATH, $((size/1024/1024)) MB)"
    else
        step "Re-downloading .nemo (existing file looks truncated, $size bytes)"
        rm -f "$NEMO_PATH"
        curl -L --fail -o "$NEMO_PATH" "$NEMO_URL"
    fi
else
    step "Downloading .nemo (~2.4 GB)"
    note "from $NEMO_URL"
    curl -L --fail -o "$NEMO_PATH" "$NEMO_URL"
fi
ok ".nemo at $NEMO_PATH"

# 2) Extract
if [[ -f "$EXTRACT_DIR/model_weights.ckpt" ]]; then
    step "Skipping extract ($EXTRACT_DIR/model_weights.ckpt exists)"
else
    step "Extracting .nemo → $EXTRACT_DIR/"
    mkdir -p "$EXTRACT_DIR"
    tar -xf "$NEMO_PATH" -C "$EXTRACT_DIR"
fi
[[ -f "$EXTRACT_DIR/model_weights.ckpt" ]] || die "extracted dir missing model_weights.ckpt"
ok "model_weights.ckpt + tokenizer + config in $EXTRACT_DIR"

# 3) Convert weights to safetensors
if [[ -f "$SAFETENSORS" ]]; then
    step "Skipping convert ($SAFETENSORS exists)"
else
    step "Converting checkpoint → safetensors (Python)"
    python3 tools/convert_nemo.py \
        --ckpt "$EXTRACT_DIR/model_weights.ckpt" \
        --out  "$SAFETENSORS"
fi
[[ -f "$SAFETENSORS" ]] || die "convert step did not produce $SAFETENSORS"
ok "safetensors at $SAFETENSORS"

# 4) Copy tokenizer to a stable path
if [[ -f "$TOKENIZER" ]]; then
    step "Skipping tokenizer copy ($TOKENIZER exists)"
else
    step "Copying SentencePiece tokenizer → $TOKENIZER"
    # The .nemo names this with a UUID prefix; glob it out.
    src=$(ls "$EXTRACT_DIR"/*tokenizer.model 2>/dev/null | head -1 || true)
    [[ -n "$src" ]] || die "no *tokenizer.model in $EXTRACT_DIR"
    cp "$src" "$TOKENIZER"
fi
ok "tokenizer at $TOKENIZER"

# 5) Done — print sizes and next command
step "Ready"
ls -lh "$NEMO_PATH" "$SAFETENSORS" "$TOKENIZER" | awk '{printf "    %-12s  %s\n", $5, $NF}'
cat <<'EOF'

Next: transcribe a wav file
    cargo run --release --features cuda --bin transcribe -- \
        --audio path/to/some.wav \
        --st  models/nemotron-speech-streaming-en-0.6b.safetensors \
        --tok models/tokenizer.model

(drop --features cuda for CPU; replace with --features metal on macOS)
EOF
