use std::{
    env,
    fs::File,
    io::BufWriter,
    path::Path,
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use reqwest::{
    blocking::Client,
    header::{ACCEPT, AUTHORIZATION, HeaderMap, USER_AGENT},
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::Language;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryManifest {
    pub discovery_seed: u64,
    pub repositories: Vec<RepositorySpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositorySpec {
    pub name: String,
    pub language: Language,
    pub clone_url: String,
    pub repository_url: String,
    pub default_branch: String,
    pub license: String,
    pub size_kb: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    items: Vec<SearchRepository>,
}

#[derive(Debug, Deserialize)]
struct SearchRepository {
    full_name: String,
    clone_url: String,
    html_url: String,
    default_branch: String,
    size: u64,
    license: Option<SearchLicense>,
}

#[derive(Debug, Deserialize)]
struct SearchLicense {
    spdx_id: String,
}

const PERMISSIVE_LICENSES: &[&str] = &[
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unlicense",
];

pub fn discover(
    repositories_per_language: usize,
    minimum_stars: u32,
    seed: u64,
    allow_unknown_license: bool,
) -> Result<RepositoryManifest> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("failed to construct GitHub HTTP client")?;
    let token = env::var("GITHUB_TOKEN").ok();
    let mut repositories = Vec::new();

    for language in Language::ALL {
        let mut found = 0;
        // GitHub search returns at most 1000 results. Ten 100-item pages is enough here.
        for page in 1..=10 {
            if found >= repositories_per_language {
                break;
            }
            let query = format!(
                "language:{} stars:>={minimum_stars} fork:false archived:false",
                language.github_name()
            );
            let page = page.to_string();
            let mut request = client
                .get("https://api.github.com/search/repositories")
                .query(&[
                    ("q", query.as_str()),
                    ("sort", "updated"),
                    ("order", "desc"),
                    ("per_page", "100"),
                    ("page", page.as_str()),
                ])
                .header(USER_AGENT, "code-classifier/0.1")
                .header(ACCEPT, "application/vnd.github+json")
                .header("X-GitHub-Api-Version", "2022-11-28");
            if let Some(token) = &token {
                request = request.header(AUTHORIZATION, format!("Bearer {token}"));
            }
            let response = request
                .send()
                .with_context(|| format!("GitHub repository search failed for {language}"))?;
            observe_rate_limit(response.headers());
            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().unwrap_or_default();
                bail!("GitHub search returned {status}: {body}");
            }
            let result: SearchResponse = response
                .json()
                .with_context(|| format!("invalid GitHub response for {language}"))?;
            if result.items.is_empty() {
                break;
            }

            for repository in result.items {
                let license = repository
                    .license
                    .map(|license| license.spdx_id)
                    .unwrap_or_else(|| "UNKNOWN".to_owned());
                if !allow_unknown_license && !PERMISSIVE_LICENSES.contains(&license.as_str()) {
                    continue;
                }
                repositories.push(RepositorySpec {
                    name: repository.full_name,
                    language,
                    clone_url: repository.clone_url,
                    repository_url: repository.html_url,
                    default_branch: repository.default_branch,
                    license,
                    size_kb: repository.size,
                    commit: None,
                    local_path: None,
                });
                found += 1;
                if found >= repositories_per_language {
                    break;
                }
            }
        }
        info!(%language, repositories = found, "repository discovery complete");
        if found < repositories_per_language {
            warn!(
                %language,
                requested = repositories_per_language,
                found,
                "fewer eligible repositories found than requested"
            );
        }
    }

    Ok(RepositoryManifest {
        discovery_seed: seed,
        repositories,
    })
}

pub fn load_manifest(path: &Path) -> Result<RepositoryManifest> {
    let file =
        File::open(path).with_context(|| format!("failed to open manifest {}", path.display()))?;
    serde_json::from_reader(file)
        .with_context(|| format!("failed to parse manifest {}", path.display()))
}

pub fn save_manifest(path: &Path, manifest: &RepositoryManifest) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)
        .with_context(|| format!("failed to create manifest {}", path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(file), manifest)
        .with_context(|| format!("failed to write manifest {}", path.display()))
}

fn observe_rate_limit(headers: &HeaderMap) {
    let remaining = headers
        .get("x-ratelimit-remaining")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let reset = headers
        .get("x-ratelimit-reset")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    info!(?remaining, ?reset, "GitHub rate limit");

    if remaining == Some(0) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let wait_seconds = reset.unwrap_or(now).saturating_sub(now) + 1;
        warn!(wait_seconds, "GitHub rate limit exhausted; backing off");
        thread::sleep(Duration::from_secs(wait_seconds));
    }
}
