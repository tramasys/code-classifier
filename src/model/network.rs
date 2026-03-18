use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{Language, features::FEATURE_COUNT, stats::Stats};

use super::{
    activation::{relu, relu_backward},
    dense::Dense,
    loss::{LossOutput, softmax, softmax_cross_entropy, softmax_cross_entropy_gradient},
};

pub const INPUT_SIZE: usize = FEATURE_COUNT;
pub const HIDDEN_1: usize = 64;
pub const HIDDEN_2: usize = 32;
pub const OUTPUT_SIZE: usize = Language::COUNT;
pub const PARAMETER_COUNT: usize = INPUT_SIZE * HIDDEN_1
    + HIDDEN_1
    + HIDDEN_1 * HIDDEN_2
    + HIDDEN_2
    + HIDDEN_2 * OUTPUT_SIZE
    + OUTPUT_SIZE;

#[derive(Debug, Clone)]
pub struct ForwardCache {
    pub z1: [f32; HIDDEN_1],
    pub a1: [f32; HIDDEN_1],
    pub z2: [f32; HIDDEN_2],
    pub a2: [f32; HIDDEN_2],
    pub logits: [f32; OUTPUT_SIZE],
    pub probabilities: [f32; OUTPUT_SIZE],
}

impl Default for ForwardCache {
    fn default() -> Self {
        Self {
            z1: [0.0; HIDDEN_1],
            a1: [0.0; HIDDEN_1],
            z2: [0.0; HIDDEN_2],
            a2: [0.0; HIDDEN_2],
            logits: [0.0; OUTPUT_SIZE],
            probabilities: [0.0; OUTPUT_SIZE],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    layer1: Dense,
    layer2: Dense,
    layer3: Dense,
}

impl Network {
    pub fn new(rng: &mut impl Rng) -> Self {
        Self {
            layer1: Dense::new(INPUT_SIZE, HIDDEN_1, rng),
            layer2: Dense::new(HIDDEN_1, HIDDEN_2, rng),
            layer3: Dense::new(HIDDEN_2, OUTPUT_SIZE, rng),
        }
    }

    pub fn forward(&self, input: &[f32; INPUT_SIZE]) -> ForwardCache {
        assert!(input.iter().all(|value| value.is_finite()));
        let mut cache = ForwardCache::default();

        // z1: [64] = W1: [64, 256] * input: [256] + b1: [64]
        self.layer1.forward(input, &mut cache.z1);
        cache.a1 = cache.z1;
        relu(&mut cache.a1);

        // z2: [32] = W2: [32, 64] * a1: [64] + b2: [32]
        self.layer2.forward(&cache.a1, &mut cache.z2);
        cache.a2 = cache.z2;
        relu(&mut cache.a2);

        // logits: [5] = W3: [5, 32] * a2: [32] + b3: [5]
        self.layer3.forward(&cache.a2, &mut cache.logits);
        cache.probabilities = softmax(&cache.logits);

        assert!(cache.probabilities.iter().all(|value| value.is_finite()));
        cache
    }

    pub fn loss(&self, input: &[f32; INPUT_SIZE], target: Language) -> LossOutput {
        let cache = self.forward(input);
        softmax_cross_entropy(&cache.logits, target)
    }

    /// Accumulates gradients for one sample. All temporary arrays expose the
    /// exact mathematical vector shape used at each backpropagation step.
    pub fn backward(&mut self, input: &[f32; INPUT_SIZE], target: Language, cache: &ForwardCache) {
        // dz3: [5] = probabilities - one_hot(target)
        let dz3 = softmax_cross_entropy_gradient(&cache.probabilities, target as usize);

        // Accumulate dW3/db3 and compute da2: [32] = W3^T * dz3.
        let mut da2 = [0.0; HIDDEN_2];
        self.layer3.backward(&cache.a2, &dz3, &mut da2);

        // dz2: [32] = da2 * ReLU'(z2)
        let mut dz2 = da2;
        relu_backward(&cache.z2, &mut dz2);

        // Accumulate dW2/db2 and compute da1: [64] = W2^T * dz2.
        let mut da1 = [0.0; HIDDEN_1];
        self.layer2.backward(&cache.a1, &dz2, &mut da1);

        // dz1: [64] = da1 * ReLU'(z1)
        let mut dz1 = da1;
        relu_backward(&cache.z1, &mut dz1);

        // Accumulate dW1/db1. d(input) is shown explicitly although training does not use it.
        let mut grad_input = [0.0; INPUT_SIZE];
        self.layer1.backward(input, &dz1, &mut grad_input);

        assert!(self.gradients_are_finite());
    }

    pub fn zero_grad(&mut self) {
        self.layer1.zero_grad();
        self.layer2.zero_grad();
        self.layer3.zero_grad();
    }

    pub fn scale_gradients(&mut self, scale: f32) {
        assert!(scale.is_finite());
        self.layer1.scale_gradients(scale);
        self.layer2.scale_gradients(scale);
        self.layer3.scale_gradients(scale);
    }

    pub fn step(&mut self, learning_rate: f32) {
        self.layer1.step(learning_rate);
        self.layer2.step(learning_rate);
        self.layer3.step(learning_rate);
        assert!(self.parameters_are_finite());
    }

    pub fn parameter_count(&self) -> usize {
        let count = self.layer1.parameter_count()
            + self.layer2.parameter_count()
            + self.layer3.parameter_count();
        debug_assert_eq!(count, PARAMETER_COUNT);
        count
    }

    pub fn weight_norms(&self) -> [f32; 3] {
        [
            Stats::from_slice(self.layer1.weights()).l2_norm,
            Stats::from_slice(self.layer2.weights()).l2_norm,
            Stats::from_slice(self.layer3.weights()).l2_norm,
        ]
    }

    pub fn gradient_norms(&self) -> [f32; 3] {
        [
            Stats::from_slice(self.layer1.grad_weights()).l2_norm,
            Stats::from_slice(self.layer2.grad_weights()).l2_norm,
            Stats::from_slice(self.layer3.grad_weights()).l2_norm,
        ]
    }

    pub fn parameters_are_finite(&self) -> bool {
        [&self.layer1, &self.layer2, &self.layer3]
            .iter()
            .all(|layer| {
                layer
                    .weights()
                    .iter()
                    .chain(layer.biases())
                    .all(|value| value.is_finite())
            })
    }

    pub fn gradients_are_finite(&self) -> bool {
        [&self.layer1, &self.layer2, &self.layer3]
            .iter()
            .all(|layer| {
                layer
                    .grad_weights()
                    .iter()
                    .chain(layer.grad_biases())
                    .all(|value| value.is_finite())
            })
    }

    pub fn validate_and_prepare(&mut self) -> Result<(), String> {
        if !self.layer1.is_valid(INPUT_SIZE, HIDDEN_1)
            || !self.layer2.is_valid(HIDDEN_1, HIDDEN_2)
            || !self.layer3.is_valid(HIDDEN_2, OUTPUT_SIZE)
        {
            return Err("checkpoint layer dimensions do not match 256 -> 64 -> 32 -> 5".to_owned());
        }
        self.layer1.ensure_gradient_storage();
        self.layer2.ensure_gradient_storage();
        self.layer3.ensure_gradient_storage();
        if !self.parameters_are_finite() {
            return Err("checkpoint contains non-finite parameters".to_owned());
        }
        Ok(())
    }

    pub(crate) fn parameter(&self, global_index: usize) -> f32 {
        let (layer, local_index) = self.parameter_location(global_index);
        layer.parameter(local_index)
    }

    pub(crate) fn set_parameter(&mut self, global_index: usize, value: f32) {
        let first_end = self.layer1.parameter_count();
        let second_end = first_end + self.layer2.parameter_count();
        if global_index < first_end {
            self.layer1.set_parameter(global_index, value);
        } else if global_index < second_end {
            self.layer2.set_parameter(global_index - first_end, value);
        } else {
            self.layer3.set_parameter(global_index - second_end, value);
        }
    }

    pub(crate) fn gradient(&self, global_index: usize) -> f32 {
        let (layer, local_index) = self.parameter_location(global_index);
        layer.gradient(local_index)
    }

    pub(crate) fn parameter_description(
        &self,
        global_index: usize,
    ) -> (&'static str, &'static str, usize) {
        let first_end = self.layer1.parameter_count();
        let second_end = first_end + self.layer2.parameter_count();
        let (name, layer, local_index) = if global_index < first_end {
            ("layer1", &self.layer1, global_index)
        } else if global_index < second_end {
            ("layer2", &self.layer2, global_index - first_end)
        } else {
            ("layer3", &self.layer3, global_index - second_end)
        };
        let (kind, index) = layer.parameter_kind_and_index(local_index);
        (name, kind, index)
    }

    fn parameter_location(&self, global_index: usize) -> (&Dense, usize) {
        assert!(global_index < self.parameter_count());
        let first_end = self.layer1.parameter_count();
        let second_end = first_end + self.layer2.parameter_count();
        if global_index < first_end {
            (&self.layer1, global_index)
        } else if global_index < second_end {
            (&self.layer2, global_index - first_end)
        } else {
            (&self.layer3, global_index - second_end)
        }
    }
}
