use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "code-classifier", version, about)]
pub struct Cli {
    #[arg(long, global = true)]
    pub log_level: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Dataset {
        #[command(subcommand)]
        command: DatasetCommand,
    },
    Gradcheck(GradcheckArgs),
    Train(TrainArgs),
    Eval(EvalArgs),
    Predict(PredictArgs),
    Inspect(PredictArgs),
}

#[derive(Debug, Subcommand)]
pub enum DatasetCommand {
    Discover(DiscoverArgs),
    Clone(CloneArgs),
    Extract(ExtractArgs),
    Stats(DatasetStatsArgs),
}

#[derive(Debug, Args)]
pub struct DiscoverArgs {
    #[arg(long, default_value_t = 100)]
    pub repositories_per_language: usize,
    #[arg(long, default_value_t = 20)]
    pub minimum_stars: u32,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value = "data/repositories.json")]
    pub output: PathBuf,
    #[arg(long)]
    pub allow_unknown_license: bool,
}

#[derive(Debug, Args)]
pub struct CloneArgs {
    #[arg(long, default_value = "data/repositories.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value = "data/repos")]
    pub repositories_directory: PathBuf,
    #[arg(long, default_value_t = 250)]
    pub maximum_size_mb: u64,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct ExtractArgs {
    #[arg(long, default_value = "data/repositories.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value = "data/repos")]
    pub repositories_directory: PathBuf,
    #[arg(long, default_value = "data")]
    pub output_directory: PathBuf,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 1_048_576)]
    pub maximum_file_bytes: u64,
    #[arg(long, default_value_t = 20)]
    pub maximum_samples_per_file: usize,
    #[arg(long, default_value_t = 500)]
    pub maximum_samples_per_repository: usize,
    #[arg(long, default_value_t = 25_000)]
    pub target_samples_per_language: usize,
}

#[derive(Debug, Args)]
pub struct DatasetStatsArgs {
    pub dataset: PathBuf,
}

#[derive(Debug, Args)]
pub struct GradcheckArgs {
    #[arg(long, default_value_t = 32)]
    pub samples: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 1e-3)]
    pub epsilon: f32,
    #[arg(long, default_value_t = 2e-3)]
    pub absolute_tolerance: f32,
    #[arg(long, default_value_t = 2e-2)]
    pub relative_tolerance: f32,
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[arg(long)]
    pub train: PathBuf,
    #[arg(long)]
    pub validation: PathBuf,
    #[arg(long)]
    pub test: Option<PathBuf>,
    #[arg(long, default_value = "models/best.json")]
    pub output: PathBuf,
    #[arg(long, default_value_t = 20)]
    pub epochs: usize,
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 0.01)]
    pub learning_rate: f32,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 100)]
    pub debug_every: usize,
}

#[derive(Debug, Args)]
pub struct EvalArgs {
    #[arg(long)]
    pub model: PathBuf,
    pub dataset: PathBuf,
}

#[derive(Debug, Args)]
pub struct PredictArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(allow_hyphen_values = true)]
    pub snippet: String,
}
