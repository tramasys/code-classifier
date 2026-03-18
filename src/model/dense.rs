use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// A fully connected layer implementing `output = weights * input + biases`.
///
/// Each contiguous row contains the weights for one output neuron. A weight at
/// `(output, input)` always has flattened index `output * input_size + input`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dense {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
    #[serde(skip)]
    grad_weights: Vec<f32>,
    #[serde(skip)]
    grad_biases: Vec<f32>,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize, rng: &mut impl Rng) -> Self {
        assert!(input_size > 0);
        assert!(output_size > 0);
        let standard_deviation = (2.0 / input_size as f32).sqrt();
        let distribution = Normal::new(0.0, standard_deviation)
            .expect("He initialization has a positive standard deviation");
        let weights = (0..input_size * output_size)
            .map(|_| distribution.sample(rng))
            .collect();

        Self {
            input_size,
            output_size,
            weights,
            biases: vec![0.0; output_size],
            grad_weights: vec![0.0; input_size * output_size],
            grad_biases: vec![0.0; output_size],
        }
    }

    pub fn from_parameters(
        input_size: usize,
        output_size: usize,
        weights: Vec<f32>,
        biases: Vec<f32>,
    ) -> Self {
        assert_eq!(weights.len(), input_size * output_size);
        assert_eq!(biases.len(), output_size);
        Self {
            input_size,
            output_size,
            weights,
            biases,
            grad_weights: vec![0.0; input_size * output_size],
            grad_biases: vec![0.0; output_size],
        }
    }

    // Indexed loops keep the row-major weight equation visible to learners.
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.input_size);
        assert_eq!(output.len(), self.output_size);

        // W has shape [output_size, input_size], so each output is one row's dot product.
        for output_index in 0..self.output_size {
            let mut value = self.biases[output_index];
            for input_index in 0..self.input_size {
                let weight_index = output_index * self.input_size + input_index;
                value += self.weights[weight_index] * input[input_index];
            }
            output[output_index] = value;
        }
    }

    /// Accumulates parameter gradients and adds `W^T * grad_output` to `grad_input`.
    #[allow(clippy::needless_range_loop)]
    pub fn backward(&mut self, input: &[f32], grad_output: &[f32], grad_input: &mut [f32]) {
        assert_eq!(input.len(), self.input_size);
        assert_eq!(grad_output.len(), self.output_size);
        assert_eq!(grad_input.len(), self.input_size);

        for output_index in 0..self.output_size {
            let output_gradient = grad_output[output_index];
            self.grad_biases[output_index] += output_gradient;
            for input_index in 0..self.input_size {
                let weight_index = output_index * self.input_size + input_index;
                self.grad_weights[weight_index] += output_gradient * input[input_index];
                grad_input[input_index] += self.weights[weight_index] * output_gradient;
            }
        }
    }

    pub fn zero_grad(&mut self) {
        self.ensure_gradient_storage();
        self.grad_weights.fill(0.0);
        self.grad_biases.fill(0.0);
    }

    pub fn scale_gradients(&mut self, scale: f32) {
        for gradient in &mut self.grad_weights {
            *gradient *= scale;
        }
        for gradient in &mut self.grad_biases {
            *gradient *= scale;
        }
    }

    pub fn step(&mut self, learning_rate: f32) {
        assert!(learning_rate.is_finite() && learning_rate >= 0.0);
        for (weight, gradient) in self.weights.iter_mut().zip(&self.grad_weights) {
            *weight -= learning_rate * gradient;
        }
        for (bias, gradient) in self.biases.iter_mut().zip(&self.grad_biases) {
            *bias -= learning_rate * gradient;
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn biases(&self) -> &[f32] {
        &self.biases
    }

    pub fn grad_weights(&self) -> &[f32] {
        &self.grad_weights
    }

    pub fn grad_biases(&self) -> &[f32] {
        &self.grad_biases
    }

    pub(crate) fn parameter(&self, index: usize) -> f32 {
        if index < self.weights.len() {
            self.weights[index]
        } else {
            self.biases[index - self.weights.len()]
        }
    }

    pub(crate) fn set_parameter(&mut self, index: usize, value: f32) {
        if index < self.weights.len() {
            self.weights[index] = value;
        } else {
            self.biases[index - self.weights.len()] = value;
        }
    }

    pub(crate) fn gradient(&self, index: usize) -> f32 {
        if index < self.grad_weights.len() {
            self.grad_weights[index]
        } else {
            self.grad_biases[index - self.grad_weights.len()]
        }
    }

    pub(crate) fn parameter_kind_and_index(&self, index: usize) -> (&'static str, usize) {
        if index < self.weights.len() {
            ("weight", index)
        } else {
            ("bias", index - self.weights.len())
        }
    }

    pub(crate) fn ensure_gradient_storage(&mut self) {
        self.grad_weights.resize(self.weights.len(), 0.0);
        self.grad_biases.resize(self.biases.len(), 0.0);
    }

    pub(crate) fn is_valid(&self, expected_input: usize, expected_output: usize) -> bool {
        self.input_size == expected_input
            && self.output_size == expected_output
            && self.weights.len() == expected_input * expected_output
            && self.biases.len() == expected_output
    }
}
