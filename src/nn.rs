use rand::Rng;

use crate::{engine::Value, operators::Num};

pub struct Neuron<T: Num> {
    pub(crate) weights: Vec<Value<T>>,
    pub(crate) bias: Value<T>,
    pub(crate) activation: fn(Value<T>) -> Value<T>,
}

impl<T: Num> Neuron<T> {
    pub fn new(num_inputs: usize, activation: fn(Value<T>) -> Value<T>) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..num_inputs)
            .map(|_| Value::from(rng.gen_range(-T::one()..T::one())))
            .collect();
        let bias = Value::from(rng.gen_range(-T::one()..T::one()));
        Neuron {
            weights,
            bias,
            activation,
        }
    }

    pub fn forward(&self, inputs: &[Value<T>]) -> Value<T> {
        let sum = self
            .weights
            .iter()
            .zip(inputs)
            .map(|(w, i)| w.clone() * i.clone())
            .fold(Value::from(T::zero()), |acc, x| acc + x);

        let activation = self.activation;
        activation(sum + self.bias.clone())
    }
}

pub struct Layer<T: Num> {
    pub(crate) neurons: Vec<Neuron<T>>,
}

impl<T: Num> Layer<T> {
    pub fn new(
        num_inputs: usize,
        num_neurons: usize,
        activation: fn(Value<T>) -> Value<T>,
    ) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Neuron::new(num_inputs, activation))
            .collect();
        Layer { neurons }
    }

    pub fn forward(&self, inputs: &[Value<T>]) -> Vec<Value<T>> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }
}

pub struct MLP<T: Num> {
    pub(crate) layers: Vec<Layer<T>>,
}

impl<T: Num> MLP<T> {
    pub fn new(sizes: &[usize], activation: fn(Value<T>) -> Value<T>) -> Self {
        let layers = sizes
            .windows(2)
            .map(|w| Layer::new(w[0], w[1], activation))
            .collect();
        MLP { layers }
    }

    pub fn forward(&self, inputs: &[Value<T>]) -> Vec<Value<T>> {
        self.layers
            .iter()
            .fold(inputs.to_vec(), |acc, layer| layer.forward(&acc))
    }

    pub fn step(&mut self, lr: T) {
        for layer in self.layers.iter_mut() {
            for neuron in layer.neurons.iter_mut() {
                for weight in neuron.weights.iter_mut() {
                    weight.step(lr);
                }
                neuron.bias.step(lr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::tanh;

    #[test]
    fn test_neuron() {
        let neuron = Neuron::new(2, tanh);
        let inputs = vec![Value::from(1.0), Value::from(2.0)];
        let output = neuron.forward(&inputs);

        let weights = neuron.weights.iter().map(|w| w.data()).collect::<Vec<_>>();
        let bias = neuron.bias.data();

        assert_eq!(weights.len(), 2);
        assert_eq!(
            output.data(),
            (weights[0] * inputs[0].data() + weights[1] * inputs[1].data() + bias).tanh()
        );
    }

    #[test]
    fn test_layer() {
        let layer = Layer::new(2, 3, tanh);
        let inputs = vec![Value::from(1.0), Value::from(2.0)];
        let outputs = layer.forward(&inputs);
        assert_eq!(outputs.len(), 3);

        for (neuron, output) in layer.neurons.iter().zip(outputs.iter()) {
            let weights = neuron.weights.iter().map(|w| w.data()).collect::<Vec<_>>();
            let bias = neuron.bias.data();

            assert_eq!(
                output.data(),
                (weights[0] * inputs[0].data() + weights[1] * inputs[1].data() + bias).tanh()
            );
        }
    }

    #[test]
    fn test_mlp() {
        let mlp = MLP::new(&[2, 3, 1], tanh);
        let inputs = vec![Value::from(1.0), Value::from(2.0)];
        let outputs = mlp.forward(&inputs);
        assert_eq!(outputs.len(), 1);

        for (layer_index, layer) in mlp.layers.iter().enumerate() {
            let layer_inputs = if layer_index == 0 {
                inputs.clone()
            } else {
                mlp.layers[layer_index - 1].forward(&inputs)
            };

            let layer_outputs = layer.forward(&layer_inputs);
            assert_eq!(layer_outputs.len(), layer.neurons.len());

            for (neuron, output) in layer.neurons.iter().zip(layer_outputs.iter()) {
                let weights = neuron.weights.iter().map(|w| w.data()).collect::<Vec<_>>();
                let bias = neuron.bias.data();

                let expected_output = weights
                    .iter()
                    .zip(layer_inputs.iter())
                    .map(|(w, i)| w * i.data())
                    .sum::<f64>()
                    + bias;

                assert_eq!(output.data(), expected_output.tanh());
            }
        }
    }
}
