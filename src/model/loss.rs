use crate::Language;

#[derive(Debug, Clone, Copy)]
pub struct LossOutput {
    pub loss: f32,
    pub probabilities: [f32; Language::COUNT],
}

pub fn softmax<const N: usize>(logits: &[f32; N]) -> [f32; N] {
    assert!(N > 0);
    assert!(logits.iter().all(|value| value.is_finite()));
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = [0.0; N];
    let mut sum = 0.0;
    for index in 0..N {
        probabilities[index] = (logits[index] - maximum).exp();
        sum += probabilities[index];
    }
    assert!(sum.is_finite() && sum > 0.0);
    for probability in &mut probabilities {
        *probability /= sum;
    }
    debug_assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    probabilities
}

pub fn cross_entropy(probabilities: &[f32], target: usize) -> f32 {
    assert!(target < probabilities.len());
    let target_probability = probabilities[target].clamp(f32::MIN_POSITIVE, 1.0);
    let loss = -target_probability.ln();
    assert!(loss.is_finite());
    loss
}

pub fn softmax_cross_entropy(logits: &[f32; Language::COUNT], target: Language) -> LossOutput {
    let probabilities = softmax(logits);
    LossOutput {
        loss: cross_entropy(&probabilities, target as usize),
        probabilities,
    }
}

/// For softmax followed by cross-entropy, the Jacobian and loss derivative
/// simplify exactly to `probabilities - one_hot(target)`.
pub fn softmax_cross_entropy_gradient<const N: usize>(
    probabilities: &[f32; N],
    target: usize,
) -> [f32; N] {
    assert!(target < N);
    let mut gradient = *probabilities;
    gradient[target] -= 1.0;
    gradient
}
