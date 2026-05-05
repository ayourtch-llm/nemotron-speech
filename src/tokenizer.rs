//! Tokenizer-side helpers.
//!
//! The model's vocab is shipped inside the .nemo tarball as both a
//! SentencePiece `.model` file and a textual vocab. For now we use the
//! `sentencepiece` crate so we get correct treatment of the leading-space
//! (▁) marker, fallbacks, and any byte-fallbacks the model might use.

use anyhow::{Context, Result};
use sentencepiece::SentencePieceProcessor;
use std::path::Path;

pub struct Tokenizer {
    sp: SentencePieceProcessor,
}

impl Tokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let sp = SentencePieceProcessor::open(path.as_ref())
            .with_context(|| format!("loading sentencepiece model {}", path.as_ref().display()))?;
        Ok(Self { sp })
    }

    /// Detokenize a sequence of BPE token IDs into text.
    pub fn detokenize(&self, ids: &[u32]) -> Result<String> {
        let ids: Vec<u32> = ids.iter().copied().collect();
        let s = self
            .sp
            .decode_piece_ids(&ids)
            .map_err(|e| anyhow::anyhow!("sp decode: {e:?}"))?;
        Ok(s)
    }
}
