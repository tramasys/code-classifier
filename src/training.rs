use std::time::{Duration, Instant};

use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use tracing::{debug, info};

use crate::{
    Language,
    dataset::DatasetRecord,
    features::extract_features,
    metrics::ClassificationMetrics,
    model::{Network, optimizer::sgd_step},
    stats::Stats,
};

#[derive(Debug, Clone)]
pub struct PreparedSample {
    pub input: [f32; 256],
    pub target: Language,
}

impl From<&DatasetRecord> for PreparedSample {
    fn from(record: &DatasetRecord) -> Self {
        Self {
            input: extract_features(&record.snippet),
            target: record.language,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub seed: u64,
    pub debug_every: usize,
}

#[derive(Debug, Clone)]
pub struct EpochResult {
    pub epoch: usize,
    pub train: ClassificationMetrics,
    pub validation: ClassificationMetrics,
    pub elapsed: Duration,
    pub samples_per_second: f64,
}

#[derive(Debug, Clone)]
pub struct TrainingOutcome {
    pub epochs: Vec<EpochResult>,
    pub best_network: Network,
    pub best_epoch: usize,
    pub best_validation: ClassificationMetrics,
}

pub fn prepare(records: &[DatasetRecord]) -> Vec<PreparedSample> {
    records.iter().map(PreparedSample::from).collect()
}

pub fn evaluate(network: &Network, samples: &[PreparedSample]) -> ClassificationMetrics {
    let mut metrics = ClassificationMetrics::default();
    for sample in samples {
        let cache = network.forward(&sample.input);
        let loss = crate::model::loss::cross_entropy(&cache.probabilities, sample.target as usize);
        metrics.record(loss, sample.target, &cache.probabilities);
    }
    metrics
}

pub fn train(
    network: &mut Network,
    training_samples: &[PreparedSample],
    validation_samples: &[PreparedSample],
    config: TrainingConfig,
) -> TrainingOutcome {
    assert!(config.epochs > 0);
    assert!(config.batch_size > 0);
    assert!(config.learning_rate > 0.0 && config.learning_rate.is_finite());
    assert!(!training_samples.is_empty());

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut indices: Vec<usize> = (0..training_samples.len()).collect();
    let mut best_network = network.clone();
    let mut best_epoch = 0;
    let mut best_validation = ClassificationMetrics::default();
    let mut best_loss = f32::INFINITY;
    let mut epochs = Vec::with_capacity(config.epochs);

    for epoch in 1..=config.epochs {
        let started = Instant::now();
        indices.shuffle(&mut rng);

        for (batch_index, batch) in indices.chunks(config.batch_size).enumerate() {
            network.zero_grad();
            let mut batch_metrics = ClassificationMetrics::default();
            let mut active_1 = 0;
            let mut active_2 = 0;
            let mut last_z1_stats = Stats::from_slice(&[]);
            let mut last_z2_stats = Stats::from_slice(&[]);

            for sample_index in batch {
                let sample = &training_samples[*sample_index];
                let cache = network.forward(&sample.input);
                let loss =
                    crate::model::loss::cross_entropy(&cache.probabilities, sample.target as usize);
                batch_metrics.record(loss, sample.target, &cache.probabilities);
                active_1 += cache.a1.iter().filter(|value| **value > 0.0).count();
                active_2 += cache.a2.iter().filter(|value| **value > 0.0).count();
                last_z1_stats = Stats::from_slice(&cache.z1);
                last_z2_stats = Stats::from_slice(&cache.z2);
                network.backward(&sample.input, sample.target, &cache);
            }

            // Use the actual size so a final partial mini-batch gets the same averaging.
            network.scale_gradients(1.0 / batch.len() as f32);
            if config.debug_every > 0 && (batch_index + 1) % config.debug_every == 0 {
                let weight_norms = network.weight_norms();
                let gradient_norms = network.gradient_norms();
                debug!(
                    epoch,
                    batch = batch_index + 1,
                    loss = batch_metrics.loss(),
                    accuracy = batch_metrics.accuracy(),
                    weight_norms = ?weight_norms,
                    gradient_norms = ?gradient_norms,
                    relu_active_1 = active_1 as f32 / (batch.len() * 64) as f32,
                    relu_active_2 = active_2 as f32 / (batch.len() * 32) as f32,
                    last_z1 = ?last_z1_stats,
                    last_z2 = ?last_z2_stats,
                    "mini-batch"
                );
            }
            sgd_step(network, config.learning_rate);
        }

        // Re-evaluate after the epoch so train and validation metrics describe one model state.
        let train_metrics = evaluate(network, training_samples);
        let validation_metrics = evaluate(network, validation_samples);
        let elapsed = started.elapsed();
        let samples_per_second = training_samples.len() as f64 / elapsed.as_secs_f64();
        info!(
            epoch,
            total_epochs = config.epochs,
            train_loss = train_metrics.loss(),
            train_accuracy = train_metrics.accuracy(),
            validation_loss = validation_metrics.loss(),
            validation_accuracy = validation_metrics.accuracy(),
            samples_per_second,
            elapsed_seconds = elapsed.as_secs_f64(),
            "epoch complete"
        );
        info!("\n{}", validation_metrics.confusion_matrix());

        if validation_metrics.loss() < best_loss {
            best_loss = validation_metrics.loss();
            best_epoch = epoch;
            best_network = network.clone();
            best_validation = validation_metrics.clone();
            info!(
                epoch,
                validation_loss = best_loss,
                "new best validation loss"
            );
        }

        epochs.push(EpochResult {
            epoch,
            train: train_metrics,
            validation: validation_metrics,
            elapsed,
            samples_per_second,
        });
    }

    TrainingOutcome {
        epochs,
        best_network,
        best_epoch,
        best_validation,
    }
}
