use std::process::Command as ProcessCommand;

use anyhow::{Context, Result, bail};
use clap::Parser;
use code_classifier::{
    Language,
    checkpoint::Checkpoint,
    dataset::{
        clone::clone_repositories,
        extract::{ExtractionConfig, extract_and_split},
        github::{discover, load_manifest, save_manifest},
        load_jsonl,
        stats::report as dataset_report,
        write_jsonl,
    },
    features::{FEATURE_COUNT, FeatureConfig, extract_features},
    gradcheck::{GradientCheckConfig, check_network},
    metrics::predicted_language,
    model::{Network, network::PARAMETER_COUNT},
    stats::Stats,
    training::{TrainingConfig, evaluate, prepare, train},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tracing::{info, trace};
use tracing_subscriber::EnvFilter;

mod cli;

use cli::{Cli, Command, DatasetCommand, PredictArgs};

fn main() -> Result<()> {
    let cli = Cli::parse();
    initialize_logging(cli.log_level.as_deref())?;
    log_startup();

    match cli.command {
        Command::Dataset { command } => run_dataset(command),
        Command::Gradcheck(arguments) => run_gradcheck(arguments),
        Command::Train(arguments) => run_train(arguments),
        Command::Eval(arguments) => run_eval(arguments),
        Command::Predict(arguments) => run_prediction(&arguments, false),
        Command::Inspect(arguments) => run_prediction(&arguments, true),
    }
}

fn initialize_logging(log_level: Option<&str>) -> Result<()> {
    let filter = match log_level {
        Some(directive) => EnvFilter::try_new(directive),
        None => EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info")),
    }
    .context("invalid logging directive")?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()
        .map_err(|error| anyhow::anyhow!("failed to initialize logging: {error}"))?;
    Ok(())
}

fn log_startup() {
    let rust_version = ProcessCommand::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned());
    info!(
        rust_version,
        architecture = "256 -> 64 -> 32 -> 5",
        parameters = PARAMETER_COUNT,
        features = ?FeatureConfig::default(),
        "code-classifier"
    );
}

fn run_dataset(command: DatasetCommand) -> Result<()> {
    match command {
        DatasetCommand::Discover(arguments) => {
            let manifest = discover(
                arguments.repositories_per_language,
                arguments.minimum_stars,
                arguments.seed,
                arguments.allow_unknown_license,
            )?;
            save_manifest(&arguments.output, &manifest)?;
            info!(
                path = %arguments.output.display(),
                repositories = manifest.repositories.len(),
                "discovery manifest saved"
            );
        }
        DatasetCommand::Clone(arguments) => {
            let mut manifest = load_manifest(&arguments.manifest)?;
            clone_repositories(
                &mut manifest,
                &arguments.repositories_directory,
                arguments.maximum_size_mb,
            )?;
            let output = arguments.output.as_deref().unwrap_or(&arguments.manifest);
            save_manifest(output, &manifest)?;
            info!(path = %output.display(), "clone provenance saved");
        }
        DatasetCommand::Extract(arguments) => {
            let manifest = load_manifest(&arguments.manifest)?;
            let (splits, report) = extract_and_split(
                &manifest,
                &arguments.repositories_directory,
                ExtractionConfig {
                    seed: arguments.seed,
                    maximum_file_bytes: arguments.maximum_file_bytes,
                    maximum_samples_per_file: arguments.maximum_samples_per_file,
                    maximum_samples_per_repository: arguments.maximum_samples_per_repository,
                    target_samples_per_language: arguments.target_samples_per_language,
                },
            )?;
            write_jsonl(
                &arguments.output_directory.join("train.jsonl"),
                &splits.train,
            )?;
            write_jsonl(
                &arguments.output_directory.join("validation.jsonl"),
                &splits.validation,
            )?;
            write_jsonl(&arguments.output_directory.join("test.jsonl"), &splits.test)?;
            info!(
                train = splits.train.len(),
                validation = splits.validation.len(),
                test = splits.test.len(),
                files_considered = report.files_considered,
                samples_before_deduplication = report.samples_before_deduplication,
                duplicates_removed = report.duplicates_removed,
                samples_after_deduplication = report.samples_after_deduplication,
                skipped = ?report.skipped,
                samples_per_language = ?report.samples_per_language,
                "dataset extraction complete"
            );
        }
        DatasetCommand::Stats(arguments) => {
            let records = load_jsonl(&arguments.dataset)?;
            print!("{}", dataset_report(&records));
        }
    }
    Ok(())
}

fn run_gradcheck(arguments: cli::GradcheckArgs) -> Result<()> {
    if arguments.samples == 0
        || !arguments.epsilon.is_finite()
        || arguments.epsilon <= 0.0
        || !arguments.absolute_tolerance.is_finite()
        || arguments.absolute_tolerance < 0.0
        || !arguments.relative_tolerance.is_finite()
        || arguments.relative_tolerance < 0.0
    {
        bail!(
            "gradient-check samples and epsilon must be positive; tolerances must be nonnegative"
        );
    }
    let report = check_network(GradientCheckConfig {
        samples: arguments.samples,
        seed: arguments.seed,
        epsilon: arguments.epsilon,
        absolute_tolerance: arguments.absolute_tolerance,
        relative_tolerance: arguments.relative_tolerance,
    });
    println!("Gradient check");
    println!("--------------");
    println!("Checked parameters  | {:>12}", report.entries.len());
    println!("Max absolute error  | {:>12.6e}", report.max_absolute_error);
    println!(
        "Mean absolute error | {:>12.6e}",
        report.mean_absolute_error
    );
    println!("Max relative error  | {:>12.6e}", report.max_relative_error);
    println!("Outside tolerance   | {:>12}", report.outside_tolerance);
    println!(
        "Result              | {:>12}",
        if report.outside_tolerance == 0 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    if report.outside_tolerance > 0 {
        bail!("gradient check failed");
    }
    Ok(())
}

fn run_train(arguments: cli::TrainArgs) -> Result<()> {
    if arguments.epochs == 0
        || arguments.batch_size == 0
        || !arguments.learning_rate.is_finite()
        || arguments.learning_rate <= 0.0
    {
        bail!("epochs, batch size, and learning rate must be positive");
    }
    info!(
        train_path = %arguments.train.display(),
        validation_path = %arguments.validation.display(),
        test_path = ?arguments.test,
        seed = arguments.seed,
        epochs = arguments.epochs,
        batch_size = arguments.batch_size,
        learning_rate = arguments.learning_rate,
        optimizer = "SGD",
        "training configuration"
    );
    let train_records = load_jsonl(&arguments.train)?;
    let validation_records = load_jsonl(&arguments.validation)?;
    if train_records.is_empty() || validation_records.is_empty() {
        bail!("training and validation datasets must both contain samples");
    }
    let training_samples = prepare(&train_records);
    let validation_samples = prepare(&validation_records);
    let class_counts = Language::ALL.map(|language| {
        (
            language.name(),
            train_records
                .iter()
                .filter(|record| record.language == language)
                .count(),
        )
    });
    info!(
        training_samples = training_samples.len(),
        validation_samples = validation_samples.len(),
        classes = ?class_counts,
        "datasets loaded and features extracted"
    );

    let mut rng = ChaCha8Rng::seed_from_u64(arguments.seed);
    let mut network = Network::new(&mut rng);
    let outcome = train(
        &mut network,
        &training_samples,
        &validation_samples,
        TrainingConfig {
            epochs: arguments.epochs,
            batch_size: arguments.batch_size,
            learning_rate: arguments.learning_rate,
            seed: arguments.seed,
            debug_every: arguments.debug_every,
        },
    );
    let checkpoint = Checkpoint::new(
        outcome.best_network,
        arguments.seed,
        outcome.best_epoch,
        outcome.best_validation.loss(),
        outcome.best_validation.accuracy(),
    );
    checkpoint.save(&arguments.output)?;
    info!(
        path = %arguments.output.display(),
        epoch = checkpoint.epoch,
        validation_loss = checkpoint.validation_loss,
        validation_accuracy = checkpoint.validation_accuracy,
        "best checkpoint saved"
    );

    if let Some(test_path) = arguments.test {
        let test = prepare(&load_jsonl(&test_path)?);
        let metrics = evaluate(&checkpoint.network, &test);
        print_evaluation("Test evaluation", &metrics);
    }
    Ok(())
}

fn run_eval(arguments: cli::EvalArgs) -> Result<()> {
    let checkpoint = Checkpoint::load(&arguments.model)?;
    let records = load_jsonl(&arguments.dataset)?;
    let metrics = evaluate(&checkpoint.network, &prepare(&records));
    print_evaluation("Evaluation", &metrics);
    Ok(())
}

fn run_prediction(arguments: &PredictArgs, inspect: bool) -> Result<()> {
    let checkpoint = Checkpoint::load(&arguments.model)?;
    let features = extract_features(&arguments.snippet);
    let cache = checkpoint.network.forward(&features);

    if inspect {
        print_inspection(&arguments.snippet, &features, &cache);
    } else {
        println!("Prediction");
        println!("----------");
        print_probabilities(&cache.probabilities);
    }
    print_prediction_result(&cache.probabilities);
    Ok(())
}

fn print_probabilities(probabilities: &[f32; Language::COUNT]) {
    println!("Language | Probability");
    println!("---------+------------");
    for language in Language::ALL {
        println!(
            "{:<8} | {:>10.2}%",
            language.name(),
            probabilities[language as usize] * 100.0
        );
    }
}

fn print_prediction_result(probabilities: &[f32; Language::COUNT]) {
    let prediction = predicted_language(probabilities);
    println!("\nResult");
    println!("------");
    println!("Language   | {prediction}");
    println!(
        "Confidence | {:.2}%",
        probabilities[prediction as usize] * 100.0
    );
}

fn print_evaluation(title: &str, metrics: &code_classifier::metrics::ClassificationMetrics) {
    println!("{title}");
    println!("{}", "-".repeat(title.len()));
    println!("Samples  | {}", metrics.samples);
    println!("Loss     | {:.4}", metrics.loss());
    println!(
        "Accuracy | {:.2}% ({}/{})",
        metrics.accuracy() * 100.0,
        metrics.correct,
        metrics.samples
    );
    println!("\n{}", metrics.confusion_matrix());
}

fn print_inspection(
    snippet: &str,
    features: &[f32; FEATURE_COUNT],
    cache: &code_classifier::model::ForwardCache,
) {
    let nonzero: Vec<_> = features
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0.0)
        .collect();
    println!("Network inspection");
    println!("------------------");
    println!("\nSnippet");
    println!("-------");
    println!("{snippet}");

    println!("\nInput features");
    println!("--------------");
    println!("Nonzero buckets | {}", nonzero.len());
    println!("Total buckets   | {FEATURE_COUNT}");

    println!("\nActivation statistics");
    println!("---------------------");
    println!("Layer   |       Min |       Max |      Mean |   L2 norm | ReLU active");
    println!("--------+-----------+-----------+-----------+-----------+------------");
    print_stats_row("Input", Stats::from_slice(features), None);
    print_stats_row(
        "Layer 1",
        Stats::from_slice(&cache.z1),
        Some((
            cache.a1.iter().filter(|value| **value > 0.0).count(),
            cache.a1.len(),
        )),
    );
    print_stats_row(
        "Layer 2",
        Stats::from_slice(&cache.z2),
        Some((
            cache.a2.iter().filter(|value| **value > 0.0).count(),
            cache.a2.len(),
        )),
    );

    println!("\nOutputs");
    println!("-------");
    println!("Language |      Logit | Probability");
    println!("---------+------------+------------");
    for language in Language::ALL {
        println!(
            "{:<8} | {:+10.5} | {:>10.2}%",
            language.name(),
            cache.logits[language as usize],
            cache.probabilities[language as usize] * 100.0
        );
    }

    trace!(snippet, feature_buckets = ?nonzero, z1 = ?cache.z1, a1 = ?cache.a1, z2 = ?cache.z2, a2 = ?cache.a2, logits = ?cache.logits, probabilities = ?cache.probabilities, "inspection details");
}

fn print_stats_row(name: &str, stats: Stats, active: Option<(usize, usize)>) {
    let active = active.map_or_else(
        || "-".to_owned(),
        |(count, total)| {
            format!(
                "{count}/{total} ({:.1}%)",
                count as f32 / total as f32 * 100.0
            )
        },
    );
    println!(
        "{name:<7} | {:+9.4} | {:+9.4} | {:+9.4} | {:>9.4} | {active:>11}",
        stats.min, stats.max, stats.mean, stats.l2_norm
    );
}
