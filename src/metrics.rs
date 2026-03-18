use std::fmt::Write;

use crate::Language;

#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    pub loss_sum: f64,
    pub correct: usize,
    pub samples: usize,
    /// Rows are actual classes and columns are predicted classes.
    pub confusion: [[usize; Language::COUNT]; Language::COUNT],
}

impl Default for ClassificationMetrics {
    fn default() -> Self {
        Self {
            loss_sum: 0.0,
            correct: 0,
            samples: 0,
            confusion: [[0; Language::COUNT]; Language::COUNT],
        }
    }
}

impl ClassificationMetrics {
    pub fn record(&mut self, loss: f32, target: Language, probabilities: &[f32; Language::COUNT]) {
        assert!(loss.is_finite());
        let predicted = predicted_language(probabilities);
        self.loss_sum += f64::from(loss);
        self.samples += 1;
        self.correct += usize::from(predicted == target);
        self.confusion[target as usize][predicted as usize] += 1;
    }

    pub fn loss(&self) -> f32 {
        if self.samples == 0 {
            0.0
        } else {
            (self.loss_sum / self.samples as f64) as f32
        }
    }

    pub fn accuracy(&self) -> f32 {
        if self.samples == 0 {
            0.0
        } else {
            self.correct as f32 / self.samples as f32
        }
    }

    pub fn confusion_matrix(&self) -> String {
        let mut output = String::from(
            "Confusion matrix\n\
             ----------------\n\
             Actual     |      C |    C++ |   Rust | Python |   Java |  Total | Accuracy\n\
             -----------+--------+--------+--------+--------+--------+--------+---------\n",
        );
        for actual in Language::ALL {
            let row = &self.confusion[actual as usize];
            let total: usize = row.iter().sum();
            let accuracy = if total == 0 {
                0.0
            } else {
                row[actual as usize] as f32 / total as f32 * 100.0
            };
            let _ = write!(output, "{:<10} |", actual.name());
            for count in row {
                let _ = write!(output, " {count:6} |");
            }
            let _ = writeln!(output, " {total:6} | {accuracy:7.2}%");
        }
        output
    }
}

pub fn predicted_language(probabilities: &[f32; Language::COUNT]) -> Language {
    assert!(probabilities.iter().all(|value| value.is_finite()));
    let mut best_index = 0;
    for index in 1..Language::COUNT {
        if probabilities[index] > probabilities[best_index] {
            best_index = index;
        }
    }
    Language::from_index(best_index).expect("best index is within the fixed language array")
}
