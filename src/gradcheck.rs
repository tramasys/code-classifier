use std::collections::HashSet;

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tracing::info;

use crate::{
    Language,
    model::{Network, network::INPUT_SIZE},
};

#[derive(Debug, Clone)]
pub struct GradientCheckEntry {
    pub layer: &'static str,
    pub parameter_kind: &'static str,
    pub parameter_index: usize,
    pub analytical: f32,
    pub numerical: f32,
    pub absolute_error: f32,
    pub relative_error: f32,
    pub outside_tolerance: bool,
}

#[derive(Debug, Clone)]
pub struct GradientCheckReport {
    pub entries: Vec<GradientCheckEntry>,
    pub max_absolute_error: f32,
    pub mean_absolute_error: f32,
    pub max_relative_error: f32,
    pub outside_tolerance: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct GradientCheckConfig {
    pub samples: usize,
    pub seed: u64,
    pub epsilon: f32,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
}

impl Default for GradientCheckConfig {
    fn default() -> Self {
        Self {
            samples: 32,
            seed: 42,
            epsilon: 1e-3,
            absolute_tolerance: 2e-3,
            relative_tolerance: 2e-2,
        }
    }
}

pub fn check_network(config: GradientCheckConfig) -> GradientCheckReport {
    assert!(config.epsilon > 0.0 && config.epsilon.is_finite());
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut network = Network::new(&mut rng);
    let mut input = [0.0; INPUT_SIZE];
    for value in &mut input {
        *value = rng.random_range(-0.25..0.25);
    }
    let target = Language::Rust;

    network.zero_grad();
    let cache = network.forward(&input);
    network.backward(&input, target, &cache);

    let requested = config.samples.min(network.parameter_count());
    let mut selected = HashSet::with_capacity(requested);
    while selected.len() < requested {
        selected.insert(rng.random_range(0..network.parameter_count()));
    }
    let mut selected: Vec<_> = selected.into_iter().collect();
    selected.sort_unstable();

    let mut entries = Vec::with_capacity(requested);
    for global_index in selected {
        let original = network.parameter(global_index);
        network.set_parameter(global_index, original + config.epsilon);
        let plus_loss = network.loss(&input, target).loss;
        network.set_parameter(global_index, original - config.epsilon);
        let minus_loss = network.loss(&input, target).loss;
        network.set_parameter(global_index, original);

        let analytical = network.gradient(global_index);
        let numerical = (plus_loss - minus_loss) / (2.0 * config.epsilon);
        let absolute_error = (analytical - numerical).abs();
        let relative_error = absolute_error / (analytical.abs() + numerical.abs()).max(1e-6);
        let outside_tolerance = absolute_error > config.absolute_tolerance
            && relative_error > config.relative_tolerance;
        let (layer, parameter_kind, parameter_index) = network.parameter_description(global_index);

        info!(
            layer,
            parameter_kind,
            parameter_index,
            analytical,
            numerical,
            absolute_error,
            relative_error,
            outside_tolerance,
            "gradient check parameter"
        );
        entries.push(GradientCheckEntry {
            layer,
            parameter_kind,
            parameter_index,
            analytical,
            numerical,
            absolute_error,
            relative_error,
            outside_tolerance,
        });
    }

    let max_absolute_error = entries
        .iter()
        .map(|entry| entry.absolute_error)
        .fold(0.0, f32::max);
    let mean_absolute_error = if entries.is_empty() {
        0.0
    } else {
        entries
            .iter()
            .map(|entry| entry.absolute_error)
            .sum::<f32>()
            / entries.len() as f32
    };
    let max_relative_error = entries
        .iter()
        .map(|entry| entry.relative_error)
        .fold(0.0, f32::max);
    let outside_tolerance = entries
        .iter()
        .filter(|entry| entry.outside_tolerance)
        .count();

    GradientCheckReport {
        entries,
        max_absolute_error,
        mean_absolute_error,
        max_relative_error,
        outside_tolerance,
    }
}
