use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

/// The order is part of the model format. Each value is its output-neuron index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[repr(usize)]
pub enum Language {
    C = 0,
    #[serde(rename = "C++")]
    Cpp = 1,
    Rust = 2,
    Python = 3,
    Java = 4,
}

impl Language {
    pub const COUNT: usize = 5;
    pub const ALL: [Self; Self::COUNT] = [Self::C, Self::Cpp, Self::Rust, Self::Python, Self::Java];

    pub const fn name(self) -> &'static str {
        match self {
            Self::C => "C",
            Self::Cpp => "C++",
            Self::Rust => "Rust",
            Self::Python => "Python",
            Self::Java => "Java",
        }
    }

    pub const fn github_name(self) -> &'static str {
        match self {
            Self::C => "C",
            Self::Cpp => "C++",
            Self::Rust => "Rust",
            Self::Python => "Python",
            Self::Java => "Java",
        }
    }

    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension.to_ascii_lowercase().as_str() {
            "c" => Some(Self::C),
            "cpp" | "cc" | "cxx" => Some(Self::Cpp),
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            "java" => Some(Self::Java),
            _ => None,
        }
    }

    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }
}

impl fmt::Display for Language {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl FromStr for Language {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "c" => Ok(Self::C),
            "c++" | "cpp" | "cxx" => Ok(Self::Cpp),
            "rust" | "rs" => Ok(Self::Rust),
            "python" | "py" => Ok(Self::Python),
            "java" => Ok(Self::Java),
            _ => Err(format!("unknown language: {value}")),
        }
    }
}
