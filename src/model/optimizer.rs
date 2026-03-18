use super::Network;

pub fn sgd_step(network: &mut Network, learning_rate: f32) {
    network.step(learning_rate);
}
