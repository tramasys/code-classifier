# code-classifier

`code-classifier` is a small neural network that recognizes C, C++, Rust,
Python, and Java snippets. The network uses flat `Vec<f32>`
weights, fixed-size activation arrays, and loops. As this was a learning exercise for me,
no ML libraries are used and the code is therefor not fast or efficient at all.

The project was mainly a way for me to study one complete learning system:

```text
code snippet
    │
    ▼
character 2/3-grams
    │
    ▼
hashed 256-D vector
    │
    ▼
Dense 256 -> 64
    │
   ReLU
    │
    ▼
Dense 64 -> 32
    │
   ReLU
    │
    ▼
Dense 32 -> 5
    │
  Softmax
    │
    ▼
C / C++ / Rust / Python / Java
```

The model has 18,693 trainable parameters. Start to read in
[`src/model/network.rs`](src/model/network.rs) for the whole forward and
backward pass, then read [`src/model/dense.rs`](src/model/dense.rs) for the
scalar layer operations.

## Input features

Feature extraction is in [`src/features.rs`](src/features.rs). It first changes
CRLF newlines to LF, then takes every two-byte and three-byte window. Case,
spaces, punctuation, and newlines are kept. Each window is assigned to one of
256 buckets with deterministic 64-bit FNV-1a:

```text
bucket = fnv1a(ngram) mod 256
x[bucket] += 1
```

No vocabulary is stored. Collisions are intentional. Finally, a nonzero vector
is divided by its L2 norm:

```text
x <- x / sqrt(sum_i x_i²)
```

This stops snippet length alone from controlling activation size while keeping
relative n-gram frequencies.

## Forward pass and loss

Each dense layer stores one contiguous row per output neuron. Its index is
always `output * input_size + input`, and its operation is:

```text
y_o = b_o + sum_i W_(o,i) x_i
```

Hidden activations use `ReLU(z) = max(0, z)`. Weights use seeded He
initialization, `W ~ N(0, sqrt(2 / fan_in))`, and biases start at zero.

Softmax subtracts the largest logit before exponentiation:

```text
p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
L   = -log(p_target)
```

The subtraction does not change the probabilities and prevents overflow for
large logits.

## Backpropagation and SGD

[`Network::backward`](src/model/network.rs) is the three-layer chain:

```text
dz3 = p - one_hot(target)
dW3 = dz3 outer a2        da2 = W3^T dz3
dz2 = da2 * ReLU'(z2)
dW2 = dz2 outer a1        da1 = W2^T dz2
dz1 = da1 * ReLU'(z1)
dW1 = dz1 outer input
```

For `y = Wx + b`, [`Dense::backward`](src/model/dense.rs) accumulates:

```text
dW_(o,i) += grad_output_o * input_i
db_o     += grad_output_o
dx_i     += W_(o,i) * grad_output_o
```

A mini-batch starts with zero gradients. Each sample adds its gradients. The
result is divided by the actual batch length, including a short last batch.
Basic SGD then performs `parameter -= learning_rate * gradient`. Training
reshuffles every epoch with the configured seed.

## Gradient checking

Before training real data, compare backpropagation with central differences:

```bash
cargo run -- gradcheck --samples 64 --seed 42
```

For selected parameters this computes:

```text
numerical gradient = (L(w + epsilon) - L(w - epsilon)) / (2 epsilon)
```

Every analytical/numerical pair is logged, followed by maximum, mean, and
relative errors.

## Preparing a dataset

The collector discovers repositories through the official GitHub REST API,
shallow-clones them, extracts snippets locally, removes exact duplicates, and
splits whole repositories.

```bash
# A smaller development collection (20 repositories per language).
cargo run -- dataset discover \
  --repositories-per-language 20 \
  --minimum-stars 20 \
  --seed 42

cargo run -- dataset clone \
  --manifest data/repositories.json \
  --repositories-directory data/repos

cargo run -- dataset extract \
  --manifest data/repositories.json \
  --repositories-directory data/repos \
  --output-directory data \
  --target-samples-per-language 2000 \
  --seed 42

cargo run -- dataset stats data/train.jsonl
```

Each accepted sample has 3-8 nonempty source lines and 32-512 bytes. Extraction
ignores common dependency/build directories, large and binary files, invalid
UTF-8, and obvious generated-file markers. Defaults cap each file at 20 samples
and each repository at 500.

Each JSONL record retains the language, repository URL, exact commit, SPDX
license, source path, and snippet. Exact snippets are SHA-256 deduplicated
globally. Repositories are shuffled independently per
language into roughly 80% train, 10% validation, and 10% test. This is important:
near-identical code from one project cannot leak across those boundaries.

## Training and evaluation

```bash
cargo run --release -- train \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --test data/test.jsonl \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.01 \
  --seed 42 \
  --output models/best.json
```

Features are computed from the loaded snippets rather than stored in the
dataset. Every epoch reports train/validation loss, accuracy, throughput, and a
5\*5 confusion matrix (actual classes are rows). The lowest-validation-loss
model is saved as readable JSON with its architecture, feature setup, stable
language-index mapping, seed, epoch, and validation metrics.

Evaluate a saved model separately:

```bash
cargo run -- eval --model models/best.json data/test.jsonl
```

## Prediction and inspection

```bash
cargo run -- predict \
  --model models/best.json \
  'std::vector<int> values;'

cargo run -- inspect \
  --model models/best.json \
  'let mut values = Vec::new();'
```

`predict` prints all five probabilities. `inspect` also prints input statistics,
both pre-activation ranges, active ReLU counts, and raw logits. Add
`--log-level trace` to include nonzero feature buckets and full activations:

```bash
cargo run -- --log-level trace inspect \
  --model models/best.json \
  'values = [value * 2 for value in input]'
```

`RUST_LOG=debug` exposes periodic batch loss, accuracy, weight/gradient norms,
activation statistics, and active-ReLU ratios. Normal `info` logs stay at the
epoch and dataset-operation level.

## Code structure

- [`src/features.rs`](src/features.rs): newline normalization and hashed n-grams
- [`src/model/dense.rs`](src/model/dense.rs): dense forward/backward and gradients
- [`src/model/loss.rs`](src/model/loss.rs): stable softmax and cross-entropy
- [`src/model/network.rs`](src/model/network.rs): explicit forward cache and backprop
- [`src/gradcheck.rs`](src/gradcheck.rs): sampled central differences
- [`src/training.rs`](src/training.rs): mini-batches, SGD, and epoch metrics
- [`src/checkpoint.rs`](src/checkpoint.rs): readable validated JSON models
- [`src/dataset`](src/dataset): discovery, cloning, extraction, splitting, and stats
