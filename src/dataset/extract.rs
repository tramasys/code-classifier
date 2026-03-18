use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use rand::{Rng, RngExt, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};
use tracing::{info, warn};
use walkdir::{DirEntry, WalkDir};

use crate::{Language, features::normalize_newlines};

use super::{
    DatasetRecord,
    clone::repository_path,
    github::RepositoryManifest,
    split::{DatasetSplits, split_by_repository},
};

const MIN_SNIPPET_BYTES: usize = 32;
const MAX_SNIPPET_BYTES: usize = 512;

#[derive(Debug, Clone, Copy)]
pub struct ExtractionConfig {
    pub seed: u64,
    pub maximum_file_bytes: u64,
    pub maximum_samples_per_file: usize,
    pub maximum_samples_per_repository: usize,
    pub target_samples_per_language: usize,
}

#[derive(Debug, Default)]
pub struct ExtractionReport {
    pub files_considered: usize,
    pub samples_before_deduplication: usize,
    pub duplicates_removed: usize,
    pub samples_after_deduplication: usize,
    pub skipped: BTreeMap<&'static str, usize>,
    pub samples_per_language: BTreeMap<Language, usize>,
}

impl ExtractionReport {
    fn skip(&mut self, reason: &'static str) {
        *self.skipped.entry(reason).or_default() += 1;
    }
}

pub fn extract_and_split(
    manifest: &RepositoryManifest,
    repositories_directory: &Path,
    config: ExtractionConfig,
) -> Result<(DatasetSplits, ExtractionReport)> {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut report = ExtractionReport::default();
    let mut records = Vec::new();
    let mut snippet_hashes = HashSet::new();
    let mut repositories: Vec<_> = manifest.repositories.iter().collect();
    repositories.sort_by_key(|repository| (repository.language, repository.name.as_str()));

    for repository in repositories {
        let language_total = report
            .samples_per_language
            .get(&repository.language)
            .copied()
            .unwrap_or_default();
        if language_total >= config.target_samples_per_language {
            continue;
        }

        let recorded_root = repository.local_path.as_deref().map(PathBuf::from);
        let root = recorded_root
            .filter(|path| path.exists())
            .unwrap_or_else(|| repository_path(&repository.name, repositories_directory));
        if !root.exists() {
            warn!(repository = repository.name, path = %root.display(), "clone is missing");
            report.skip("missing repository");
            continue;
        }
        let Some(commit) = &repository.commit else {
            warn!(
                repository = repository.name,
                "manifest has no resolved commit"
            );
            report.skip("missing commit");
            continue;
        };

        let mut paths: Vec<_> = WalkDir::new(&root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|entry| !is_ignored_directory(entry))
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .map(DirEntry::into_path)
            .collect();
        paths.sort();
        let mut repository_samples = 0;

        for path in paths {
            if repository_samples >= config.maximum_samples_per_repository {
                break;
            }
            let extension = path.extension().and_then(|value| value.to_str());
            if extension.and_then(Language::from_extension) != Some(repository.language) {
                report.skip("wrong extension");
                continue;
            }
            report.files_considered += 1;
            let metadata = fs::metadata(&path)
                .with_context(|| format!("failed to inspect source file {}", path.display()))?;
            if metadata.len() > config.maximum_file_bytes {
                report.skip("too large");
                continue;
            }
            let bytes = fs::read(&path)
                .with_context(|| format!("failed to read source file {}", path.display()))?;
            if bytes.contains(&0) {
                report.skip("binary");
                continue;
            }
            let Ok(source) = String::from_utf8(bytes) else {
                report.skip("invalid UTF-8");
                continue;
            };
            if source.trim().is_empty() {
                report.skip("empty");
                continue;
            }
            if looks_generated(&source) {
                report.skip("generated");
                continue;
            }

            let remaining_repository = config
                .maximum_samples_per_repository
                .saturating_sub(repository_samples);
            let current_language = report
                .samples_per_language
                .get(&repository.language)
                .copied()
                .unwrap_or_default();
            let remaining_language = config
                .target_samples_per_language
                .saturating_sub(current_language);
            let file_cap = config
                .maximum_samples_per_file
                .min(remaining_repository)
                .min(remaining_language);
            let snippets = sample_snippets(&source, file_cap, &mut rng);
            let relative_path = path.strip_prefix(&root).unwrap_or(&path);

            for snippet in snippets {
                report.samples_before_deduplication += 1;
                let hash: [u8; 32] = Sha256::digest(snippet.as_bytes()).into();
                if !snippet_hashes.insert(hash) {
                    report.duplicates_removed += 1;
                    continue;
                }
                records.push(DatasetRecord {
                    language: repository.language,
                    repository: repository.name.clone(),
                    repository_url: repository.repository_url.clone(),
                    commit: commit.clone(),
                    license: repository.license.clone(),
                    path: relative_path.to_string_lossy().into_owned(),
                    snippet,
                });
                repository_samples += 1;
                *report
                    .samples_per_language
                    .entry(repository.language)
                    .or_default() += 1;
            }
        }
        info!(
            repository = repository.name,
            language = %repository.language,
            samples = repository_samples,
            "repository extraction complete"
        );
    }

    report.samples_after_deduplication = records.len();
    let splits = split_by_repository(records, config.seed);
    Ok((splits, report))
}

fn sample_snippets(source: &str, cap: usize, rng: &mut impl Rng) -> Vec<String> {
    let normalized = normalize_newlines(source);
    let lines: Vec<_> = normalized.split_inclusive('\n').collect();
    let mut starts: Vec<_> = (0..lines.len())
        .filter(|index| !lines[*index].trim().is_empty())
        .collect();
    starts.shuffle(rng);
    let mut snippets = Vec::new();

    for start in starts {
        if snippets.len() >= cap {
            break;
        }
        let desired_nonempty = rng.random_range(3..=8);
        let mut nonempty = 0;
        let mut end = start;
        while end < lines.len() && nonempty < desired_nonempty {
            nonempty += usize::from(!lines[end].trim().is_empty());
            end += 1;
        }
        if nonempty < 3 {
            continue;
        }
        let snippet = lines[start..end].concat();
        let snippet = snippet.trim_end_matches('\n');
        if (MIN_SNIPPET_BYTES..=MAX_SNIPPET_BYTES).contains(&snippet.len()) {
            snippets.push(snippet.to_owned());
        }
    }
    snippets
}

fn is_ignored_directory(entry: &DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }
    matches!(
        entry.file_name().to_str(),
        Some(
            ".git"
                | "target"
                | "build"
                | "dist"
                | "node_modules"
                | "vendor"
                | "vendors"
                | "third_party"
                | "third-party"
                | "generated"
        )
    )
}

fn looks_generated(source: &str) -> bool {
    let prefix = source
        .chars()
        .take(4096)
        .collect::<String>()
        .to_ascii_lowercase();
    ["generated file", "automatically generated", "do not edit"]
        .iter()
        .any(|marker| prefix.contains(marker))
}
