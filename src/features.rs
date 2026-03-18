use std::borrow::Cow;

use serde::{Deserialize, Serialize};

pub const FEATURE_COUNT: usize = 256;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FeatureConfig {
    pub dimension: usize,
    pub ngram_sizes: Vec<usize>,
    pub hash: String,
    pub normalization: String,
    pub newline_normalization: String,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            dimension: FEATURE_COUNT,
            ngram_sizes: vec![2, 3],
            hash: "FNV-1a 64-bit".to_owned(),
            normalization: "L2".to_owned(),
            newline_normalization: "CRLF to LF".to_owned(),
        }
    }
}

pub fn normalize_newlines(text: &str) -> Cow<'_, str> {
    if text.contains("\r\n") {
        Cow::Owned(text.replace("\r\n", "\n"))
    } else {
        Cow::Borrowed(text)
    }
}

/// Counts hashed byte 2-grams and 3-grams, then L2-normalizes the result.
///
/// Normalization keeps long snippets from having proportionally larger layer
/// activations than short snippets. It preserves the relative n-gram frequencies.
pub fn extract_features(snippet: &str) -> [f32; FEATURE_COUNT] {
    let normalized = normalize_newlines(snippet);
    let bytes = normalized.as_bytes();
    let mut features = [0.0; FEATURE_COUNT];

    for ngram_size in [2, 3] {
        for ngram in bytes.windows(ngram_size) {
            let bucket = fnv1a(ngram) as usize % FEATURE_COUNT;
            features[bucket] += 1.0;
        }
    }

    let squared_norm: f32 = features.iter().map(|value| value * value).sum();
    if squared_norm > 0.0 {
        let norm = squared_norm.sqrt();
        for value in &mut features {
            *value /= norm;
        }
    }

    debug_assert!(features.iter().all(|value| value.is_finite()));
    features
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
