use std::collections::{BTreeMap, HashSet};

use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use crate::Language;

use super::DatasetRecord;

#[derive(Debug, Default)]
pub struct DatasetSplits {
    pub train: Vec<DatasetRecord>,
    pub validation: Vec<DatasetRecord>,
    pub test: Vec<DatasetRecord>,
}

pub fn split_by_repository(records: Vec<DatasetRecord>, seed: u64) -> DatasetSplits {
    let mut grouped: BTreeMap<Language, BTreeMap<String, Vec<DatasetRecord>>> = BTreeMap::new();
    for record in records {
        grouped
            .entry(record.language)
            .or_default()
            .entry(record.repository.clone())
            .or_default()
            .push(record);
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut splits = DatasetSplits::default();
    for language in Language::ALL {
        let mut repositories: Vec<_> = grouped
            .remove(&language)
            .unwrap_or_default()
            .into_values()
            .collect();
        repositories.shuffle(&mut rng);
        let repository_count = repositories.len();
        let (train_count, validation_count) = split_counts(repository_count);

        for (index, mut repository_records) in repositories.into_iter().enumerate() {
            if index < train_count {
                splits.train.append(&mut repository_records);
            } else if index < train_count + validation_count {
                splits.validation.append(&mut repository_records);
            } else {
                splits.test.append(&mut repository_records);
            }
        }
    }
    assert_disjoint_repositories(&splits);
    splits
}

fn split_counts(repository_count: usize) -> (usize, usize) {
    match repository_count {
        0 => (0, 0),
        1 => (1, 0),
        2 => (1, 0),
        count => {
            let train = ((count as f32 * 0.8).round() as usize).clamp(1, count - 2);
            let remaining = count - train;
            let validation = (remaining / 2).max(1);
            (train, validation)
        }
    }
}

pub fn assert_disjoint_repositories(splits: &DatasetSplits) {
    let repositories = |records: &[DatasetRecord]| {
        records
            .iter()
            .map(|record| record.repository.clone())
            .collect::<HashSet<_>>()
    };
    let train = repositories(&splits.train);
    let validation = repositories(&splits.validation);
    let test = repositories(&splits.test);
    assert!(train.is_disjoint(&validation));
    assert!(train.is_disjoint(&test));
    assert!(validation.is_disjoint(&test));
}
