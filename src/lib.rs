pub mod engine;
pub mod nn;
pub mod operators;

#[cfg(test)]
mod tests {
    use crate::engine::Value;
    use crate::nn::MLP;
    use crate::operators::tanh;

    #[test]
    fn test_mlp_training() {
        let mut mlp = MLP::new(&[2, 3, 1], tanh);

        let inputs = vec![
            vec![Value::from(0.0), Value::from(0.0)],
            vec![Value::from(0.0), Value::from(1.0)],
            vec![Value::from(1.0), Value::from(0.0)],
            vec![Value::from(1.0), Value::from(1.0)],
        ];
        let targets = vec![
            Value::from(0.0),
            Value::from(1.0),
            Value::from(1.0),
            Value::from(0.0),
        ];

        for _ in 0..2000 {
            let mut loss = Value::from(0.0);
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = mlp.forward(input)[0].clone();
                let diff = output - target.clone();
                loss = loss + diff.clone() * diff;
            }
            loss.backward();
            mlp.step(0.15);
            loss.zero_grads();
        }

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = mlp.forward(input);
            let diff: f64 = output[0].data() - target.data();
            assert!(diff.abs() < 0.1);
        }
    }
}
