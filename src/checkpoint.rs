use std::{
    fs::{self, File},
    io::{BufReader, BufWriter},
    path::Path,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::{
    Language,
    features::{FEATURE_COUNT, FeatureConfig},
    model::{
        Network,
        network::{HIDDEN_1, HIDDEN_2, OUTPUT_SIZE},
    },
};

const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureMetadata {
    pub input_size: usize,
    pub hidden_1: usize,
    pub hidden_2: usize,
    pub output_size: usize,
    pub hidden_activation: String,
    pub output_activation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub format_version: u32,
    pub architecture: ArchitectureMetadata,
    pub feature_extractor: FeatureConfig,
    pub language_index_mapping: Vec<String>,
    pub network: Network,
    pub training_seed: u64,
    pub epoch: usize,
    pub validation_loss: f32,
    pub validation_accuracy: f32,
}

impl Checkpoint {
    pub fn new(
        network: Network,
        training_seed: u64,
        epoch: usize,
        validation_loss: f32,
        validation_accuracy: f32,
    ) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            architecture: ArchitectureMetadata {
                input_size: FEATURE_COUNT,
                hidden_1: HIDDEN_1,
                hidden_2: HIDDEN_2,
                output_size: OUTPUT_SIZE,
                hidden_activation: "ReLU".to_owned(),
                output_activation: "Softmax".to_owned(),
            },
            feature_extractor: FeatureConfig::default(),
            language_index_mapping: Language::ALL
                .iter()
                .map(|language| language.name().to_owned())
                .collect(),
            network,
            training_seed,
            epoch,
            validation_loss,
            validation_accuracy,
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create checkpoint directory {}", parent.display())
            })?;
        }
        let file = File::create(path)
            .with_context(|| format!("failed to create checkpoint {}", path.display()))?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)
            .with_context(|| format!("failed to serialize checkpoint {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open checkpoint {}", path.display()))?;
        let mut checkpoint: Self = serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("failed to parse checkpoint {}", path.display()))?;
        checkpoint.validate()?;
        checkpoint
            .network
            .validate_and_prepare()
            .map_err(anyhow::Error::msg)?;
        Ok(checkpoint)
    }

    fn validate(&self) -> Result<()> {
        if self.format_version != FORMAT_VERSION {
            bail!(
                "unsupported checkpoint format version {} (expected {FORMAT_VERSION})",
                self.format_version
            );
        }
        let expected_mapping: Vec<_> = Language::ALL
            .iter()
            .map(|language| language.name().to_owned())
            .collect();
        if self.language_index_mapping != expected_mapping {
            bail!("checkpoint language-index mapping does not match this program");
        }
        if self.feature_extractor != FeatureConfig::default() {
            bail!("checkpoint feature configuration does not match this program");
        }
        if self.architecture.input_size != FEATURE_COUNT
            || self.architecture.hidden_1 != HIDDEN_1
            || self.architecture.hidden_2 != HIDDEN_2
            || self.architecture.output_size != OUTPUT_SIZE
            || self.architecture.hidden_activation != "ReLU"
            || self.architecture.output_activation != "Softmax"
        {
            bail!("checkpoint architecture metadata does not match this program");
        }
        Ok(())
    }
}
