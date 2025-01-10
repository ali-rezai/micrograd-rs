pub mod allocator;
pub mod engine;
pub mod nn;
pub mod operators;

#[cfg(test)]
mod tests {
    use crate::allocator::Allocator;
    use crate::nn::MLP;
    use crate::operators::tanh;

    #[test]
    fn test_mlp_training() {
        let mut allocator = Allocator::new();
        let mut mlp = MLP::new(&mut allocator, &[2, 3, 1], Some(tanh));

        let inputs = vec![
            vec![allocator.alloc(0.0), allocator.alloc(0.0)],
            vec![allocator.alloc(0.0), allocator.alloc(1.0)],
            vec![allocator.alloc(1.0), allocator.alloc(0.0)],
            vec![allocator.alloc(1.0), allocator.alloc(1.0)],
        ];
        let targets = vec![
            allocator.alloc(0.0),
            allocator.alloc(1.0),
            allocator.alloc(1.0),
            allocator.alloc(0.0),
        ];

        for _ in 0..2000 {
            let mut loss = allocator.alloc_t(0.0);
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = mlp.forward(input)[0].clone();
                let diff = output - *target;
                loss = loss + diff.clone() * diff;
            }
            allocator.backward();
            mlp.step(0.15);
            allocator.clear_temps();
        }

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = mlp.forward(input);
            let diff: f64 = allocator.get(output[0]).data - allocator.get(*target).data;
            assert!(diff.abs() < 0.3);
        }
    }
}
