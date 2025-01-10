use rand::Rng;

use crate::{
    allocator::{Allocator, ValueId},
    operators::Num,
};

pub struct Neuron<T: Num> {
    pub(crate) weights: Vec<ValueId<T>>,
    pub(crate) bias: ValueId<T>,
    pub(crate) activation: Option<fn(ValueId<T>) -> ValueId<T>>,
}

impl<T: Num> Neuron<T> {
    pub fn new(
        allocator: &mut Allocator<T>,
        num_inputs: usize,
        activation: Option<fn(ValueId<T>) -> ValueId<T>>,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..num_inputs)
            .map(|_| allocator.alloc(rng.gen_range(-T::one()..T::one())))
            .collect();
        let bias = allocator.alloc(rng.gen_range(-T::one()..T::one()));
        Neuron {
            weights,
            bias,
            activation,
        }
    }

    pub fn forward(&self, inputs: &[ValueId<T>]) -> ValueId<T> {
        let sum = self
            .weights
            .iter()
            .zip(inputs)
            .map(|(w, i)| *w * *i)
            .fold(self.bias, |acc, x| acc + x);

        if let Some(activation) = self.activation {
            activation(sum)
        } else {
            sum
        }
    }
}

pub struct Layer<T: Num> {
    pub(crate) neurons: Vec<Neuron<T>>,
}

impl<T: Num> Layer<T> {
    pub fn new(
        allocator: &mut Allocator<T>,
        num_inputs: usize,
        num_neurons: usize,
        activation: Option<fn(ValueId<T>) -> ValueId<T>>,
    ) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Neuron::new(allocator, num_inputs, activation))
            .collect();
        Layer { neurons }
    }

    pub fn forward(&self, inputs: &[ValueId<T>]) -> Vec<ValueId<T>> {
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
    pub fn new(
        allocator: &mut Allocator<T>,
        sizes: &[usize],
        activation: Option<fn(ValueId<T>) -> ValueId<T>>,
    ) -> Self {
        let layers = sizes
            .windows(2)
            .map(|w| Layer::new(allocator, w[0], w[1], activation))
            .collect();
        MLP { layers }
    }

    pub fn forward(&self, inputs: &[ValueId<T>]) -> Vec<ValueId<T>> {
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
    use std::f64::EPSILON;

    use super::*;
    use crate::operators::tanh;

    #[test]
    fn test_neuron() {
        let mut allocator = Allocator::new();
        let neuron = Neuron::new(&mut allocator, 2, Some(tanh));
        let inputs = vec![allocator.alloc(1.0), allocator.alloc(2.0)];
        let output = neuron.forward(&inputs);

        let weights = neuron
            .weights
            .iter()
            .map(|w| allocator.get(*w).data)
            .collect::<Vec<_>>();
        let bias = allocator.get(neuron.bias).data;

        assert_eq!(weights.len(), 2);
        assert_eq!(
            allocator.get(output).data,
            (weights[0] * allocator.get(inputs[0]).data
                + weights[1] * allocator.get(inputs[1]).data
                + bias)
                .tanh()
        );
    }

    #[test]
    fn test_layer() {
        let mut allocator = Allocator::new();
        let layer = Layer::new(&mut allocator, 2, 3, Some(tanh));
        let inputs = vec![allocator.alloc(1.0), allocator.alloc(2.0)];
        let outputs = layer.forward(&inputs);
        assert_eq!(outputs.len(), 3);

        for (neuron, output) in layer.neurons.iter().zip(outputs.iter()) {
            let weights = neuron
                .weights
                .iter()
                .map(|w| allocator.get(*w).data)
                .collect::<Vec<_>>();
            let bias = allocator.get(neuron.bias).data;

            assert_eq!(
                allocator.get(*output).data,
                (weights[0] * allocator.get(inputs[0]).data
                    + weights[1] * allocator.get(inputs[1]).data
                    + bias)
                    .tanh()
            );
        }
    }

    #[test]
    fn test_mlp() {
        let mut allocator = Allocator::new();
        let mlp = MLP::new(&mut allocator, &[2, 3, 1], Some(tanh));
        let inputs = vec![allocator.alloc(1.0), allocator.alloc(2.0)];
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
                let weights = neuron
                    .weights
                    .iter()
                    .map(|w| allocator.get(*w).data)
                    .collect::<Vec<_>>();
                let bias = allocator.get(neuron.bias).data;

                let expected_output = weights
                    .iter()
                    .zip(layer_inputs.iter())
                    .map(|(w, i)| *w * allocator.get(*i).data)
                    .sum::<f64>()
                    + bias;

                assert!(allocator.get(*output).data - expected_output.tanh() <= EPSILON);
            }
        }
    }
}
