# micrograd-rs

This project is based on [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy.

## MLP Neural Network Training Example

This example shows how to build and train a simple MLP neural network to solve the XOR problem using `micrograd-rs`.

```rust
use micrograd_rs::{engine::Value, nn::MLP, operators::tanh};

fn main() {
    // Create a simple MLP with 2 input neurons, one hidden layer with 3 neurons, and 1 output neuron
    let mut mlp = MLP::new(&[2, 3, 1], tanh);

    // Training data for the XOR problem
    let inputs = [
        vec![Value::from(0.0), Value::from(0.0)],
        vec![Value::from(0.0), Value::from(1.0)],
        vec![Value::from(1.0), Value::from(0.0)],
        vec![Value::from(1.0), Value::from(1.0)],
    ];
    let targets = [
        Value::from(0.0),
        Value::from(1.0),
        Value::from(1.0),
        Value::from(0.0),
    ];

    // Training loop
    for epoch in 1..1001 {
        let mut loss = Value::from(0.0);

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = mlp.forward(input)[0].clone();
            let diff = output - target.clone();
            loss = loss + diff.clone() * diff;
        }

        loss.backward();
        mlp.step(0.15);
        loss.zero_grads();

        if epoch % 100 == 0 || epoch == 1 {
            println!("Epoch {:>4}: Loss = {}", epoch, loss.data());
        }
    }

    // Test the trained MLP
    for input in inputs.iter() {
        let output = mlp.forward(input)[0].clone();
        println!("Input: {:?}, Output: {}", input, output.data());
    }
}
```

## Creating Custom Operators

You can create custom operators by implementing a "forward" and "backward" function for the operator.

```rust
use micrograd_rs::engine::Value;

fn sqrt(this: Value<f64>) -> Value<f64> {
    assert!(
        this.data() >= 0.0,
        "Cannot take the square root of a negative number"
    );
    let result = this.data().sqrt();
    Value::new(result, sqrt_backward, vec![this])
}

// base_grad: Gradient of the parent
// base_val: Value of the parent
fn sqrt_backward(base_grad: f64, base_val: f64, children: &mut [Value<f64>]) {
    // Derivative of sqrt(x) is 0.5 / sqrt(x)
    // In this case, since the operator is basically parent = sqrt(child), we can directly use base_val
    children[0].add_grad(base_grad * 0.5 / base_val);
}

fn main() {
    let a = Value::from(16.0);
    let b = sqrt(a.clone());
    b.backward();
    println!("Result: {}, Gradient: {}", b.data(), a.grad());
}
```
