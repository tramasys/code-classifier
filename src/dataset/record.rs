use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::Language;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetRecord {
    pub language: Language,
    pub repository: String,
    pub repository_url: String,
    pub commit: String,
    pub license: String,
    pub path: String,
    pub snippet: String,
}

pub fn load_jsonl(path: &Path) -> Result<Vec<DatasetRecord>> {
    let file =
        File::open(path).with_context(|| format!("failed to open dataset {}", path.display()))?;
    let mut records = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from {}",
                line_index + 1,
                path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str(&line).with_context(|| {
            format!(
                "invalid JSON on line {} of {}",
                line_index + 1,
                path.display()
            )
        })?;
        records.push(record);
    }
    Ok(records)
}

pub fn write_jsonl(path: &Path, records: &[DatasetRecord]) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let file = File::create(path)
        .with_context(|| format!("failed to create dataset {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}
