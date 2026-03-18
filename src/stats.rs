#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub l2_norm: f32,
}

impl Stats {
    pub fn from_slice(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                l2_norm: 0.0,
            };
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0;
        let mut sum_squares = 0.0;
        for value in values {
            min = min.min(*value);
            max = max.max(*value);
            sum += value;
            sum_squares += value * value;
        }
        Self {
            min,
            max,
            mean: sum / values.len() as f32,
            l2_norm: sum_squares.sqrt(),
        }
    }
}
