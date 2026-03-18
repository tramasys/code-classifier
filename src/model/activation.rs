pub fn relu(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

/// Multiplies an incoming gradient by ReLU's derivative at its pre-activation.
pub fn relu_backward(pre_activation: &[f32], gradient: &mut [f32]) {
    assert_eq!(pre_activation.len(), gradient.len());
    for index in 0..gradient.len() {
        if pre_activation[index] <= 0.0 {
            gradient[index] = 0.0;
        }
    }
}
