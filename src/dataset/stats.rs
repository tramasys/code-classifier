use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fmt::Write,
};

use sha2::{Digest, Sha256};

use crate::Language;

use super::DatasetRecord;

pub fn report(records: &[DatasetRecord]) -> String {
    let mut output = String::new();
    let mut repositories: BTreeMap<Language, BTreeSet<&str>> = BTreeMap::new();
    let mut files: BTreeMap<Language, BTreeSet<(&str, &str)>> = BTreeMap::new();
    let mut samples: BTreeMap<Language, usize> = BTreeMap::new();
    let mut contributions: BTreeMap<Language, BTreeMap<&str, usize>> = BTreeMap::new();
    let mut hashes = HashSet::new();
    let mut duplicates = 0;
    let mut total_bytes = 0;
    let mut total_lines = 0;
    let mut min_bytes = usize::MAX;
    let mut max_bytes = 0;

    for record in records {
        repositories
            .entry(record.language)
            .or_default()
            .insert(&record.repository);
        files
            .entry(record.language)
            .or_default()
            .insert((&record.repository, &record.path));
        *samples.entry(record.language).or_default() += 1;
        *contributions
            .entry(record.language)
            .or_default()
            .entry(&record.repository)
            .or_default() += 1;
        let length = record.snippet.len();
        total_bytes += length;
        total_lines += record.snippet.lines().count();
        min_bytes = min_bytes.min(length);
        max_bytes = max_bytes.max(length);
        let hash: [u8; 32] = Sha256::digest(record.snippet.as_bytes()).into();
        duplicates += usize::from(!hashes.insert(hash));
    }

    let count = records.len();
    output.push_str(
        "Dataset summary\n\
         ---------------\n\
         Metric                |      Value\n\
         ----------------------+-----------\n",
    );
    let _ = writeln!(output, "Samples               | {count:>10}");
    let _ = writeln!(output, "Exact duplicates      | {duplicates:>10}");
    let _ = writeln!(
        output,
        "Snippet bytes minimum | {:>10}",
        if count == 0 { 0 } else { min_bytes },
    );
    let _ = writeln!(
        output,
        "Snippet bytes mean    | {:>10.1}",
        total_bytes as f64 / count.max(1) as f64,
    );
    let _ = writeln!(output, "Snippet bytes maximum | {max_bytes:>10}");
    let _ = writeln!(
        output,
        "Mean lines per sample  | {:>10.2}",
        total_lines as f64 / count.max(1) as f64
    );

    output.push_str(
        "\nLanguage breakdown\n\
         ------------------\n\
         Language | Samples | Repositories | Files\n\
         ---------+---------+--------------+------\n",
    );
    for language in Language::ALL {
        let _ = writeln!(
            output,
            "{:<8} | {:>7} | {:>12} | {:>5}",
            language.name(),
            samples.get(&language).copied().unwrap_or_default(),
            repositories.get(&language).map_or(0, BTreeSet::len),
            files.get(&language).map_or(0, BTreeSet::len),
        );
    }

    output.push_str("\nTop repository contributions\n----------------------------\n");
    for language in Language::ALL {
        let _ = writeln!(output, "\n{language}");
        output.push_str("Repository                               | Samples\n");
        output.push_str("-----------------------------------------+--------\n");
        let mut ranked: Vec<_> = contributions
            .get(&language)
            .into_iter()
            .flat_map(|values| values.iter())
            .collect();
        ranked.sort_by_key(|(repository, count)| (std::cmp::Reverse(**count), *repository));
        for (repository, count) in ranked.into_iter().take(5) {
            let _ = writeln!(output, "{repository:<40} | {count:>7}");
        }
    }
    output
}
