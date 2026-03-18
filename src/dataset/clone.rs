use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail};
use tracing::{info, warn};

use super::github::RepositoryManifest;

pub fn clone_repositories(
    manifest: &mut RepositoryManifest,
    repositories_directory: &Path,
    maximum_size_mb: u64,
) -> Result<()> {
    fs::create_dir_all(repositories_directory).with_context(|| {
        format!(
            "failed to create repository directory {}",
            repositories_directory.display()
        )
    })?;

    for repository in &mut manifest.repositories {
        if repository.size_kb > maximum_size_mb * 1024 {
            warn!(
                repository = repository.name,
                size_kb = repository.size_kb,
                "skipping repository above size cap"
            );
            continue;
        }

        let destination = repositories_directory.join(safe_directory_name(&repository.name));
        if !destination.exists() {
            info!(repository = repository.name, "cloning repository");
            let status = Command::new("git")
                .args(["clone", "--depth", "1", "--single-branch"])
                .arg(&repository.clone_url)
                .arg(&destination)
                .status()
                .with_context(|| format!("failed to start git for {}", repository.name))?;
            if !status.success() {
                warn!(repository = repository.name, %status, "git clone failed");
                continue;
            }
        } else {
            info!(repository = repository.name, "using existing clone");
        }

        if let Some(expected_commit) = repository.commit.as_deref() {
            checkout_recorded_commit(&destination, expected_commit, &repository.name)?;
        }

        let output = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&destination)
            .output()
            .with_context(|| format!("failed to resolve commit for {}", repository.name))?;
        if !output.status.success() {
            bail!("git rev-parse failed for {}", repository.name);
        }
        repository.commit = Some(String::from_utf8(output.stdout)?.trim().to_owned());
        repository.local_path = Some(destination.to_string_lossy().into_owned());
    }
    Ok(())
}

fn checkout_recorded_commit(directory: &Path, commit: &str, repository_name: &str) -> Result<()> {
    let head = git_output(directory, &["rev-parse", "HEAD"])?;
    if head == commit {
        return Ok(());
    }
    let changes = git_output(directory, &["status", "--porcelain"])?;
    if !changes.is_empty() {
        bail!(
            "refusing to change dirty dataset clone {} while restoring commit {commit}",
            repository_name
        );
    }

    let commit_object = format!("{commit}^{{commit}}");
    let known = Command::new("git")
        .args(["cat-file", "-e", &commit_object])
        .current_dir(directory)
        .status()?
        .success();
    if !known {
        let status = Command::new("git")
            .args(["fetch", "--depth", "1", "origin", commit])
            .current_dir(directory)
            .status()?;
        if !status.success() {
            bail!("failed to fetch recorded commit {commit} for {repository_name}");
        }
    }
    let status = Command::new("git")
        .args(["checkout", "--detach", commit])
        .current_dir(directory)
        .status()?;
    if !status.success() {
        bail!("failed to check out recorded commit {commit} for {repository_name}");
    }
    Ok(())
}

fn git_output(directory: &Path, arguments: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(directory)
        .output()?;
    if !output.status.success() {
        bail!(
            "git {} failed in {}",
            arguments.join(" "),
            directory.display()
        );
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

pub fn repository_path(repository_name: &str, repositories_directory: &Path) -> PathBuf {
    repositories_directory.join(safe_directory_name(repository_name))
}

fn safe_directory_name(repository_name: &str) -> String {
    repository_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect()
}
