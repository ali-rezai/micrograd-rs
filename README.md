# micrograd-rs

This project is based on [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy.

## MLP Neural Network Training Example

This example shows how to build and train a simple MLP neural network to solve the XOR problem using `micrograd-rs`.

```rust
use micrograd_rs::{allocator::Allocator, nn::MLP, operators::tanh};

fn main() {
    let mut allocator = Allocator::new();

    // Create a simple MLP with 2 input neurons, one hidden layer with 3 neurons, and 1 output neuron
    let mut mlp = MLP::new(&mut allocator, &[2, 3, 1], Some(tanh));

    // Training data for the XOR problem
    let inputs = [
        vec![allocator.alloc(0.0), allocator.alloc(0.0)],
        vec![allocator.alloc(0.0), allocator.alloc(1.0)],
        vec![allocator.alloc(1.0), allocator.alloc(0.0)],
        vec![allocator.alloc(1.0), allocator.alloc(1.0)],
    ];
    let targets = [
        allocator.alloc(0.0),
        allocator.alloc(1.0),
        allocator.alloc(1.0),
        allocator.alloc(0.0),
    ];

    // Training loop
    for epoch in 1..1001 {
        let mut loss = allocator.alloc_t(0.0);

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = mlp.forward(input)[0];
            let diff = output - *target;
            loss = loss + diff * diff;
        }

        allocator.backward();
        mlp.step(0.15);
        let loss_data = allocator.get(loss).data;
        allocator.zero_grads();

        if epoch % 100 == 0 || epoch == 1 {
            println!("Epoch {:>4}: Loss = {}", epoch, loss_data);
        }
    }

    // Test the trained MLP
    for input in inputs.iter() {
        let output = mlp.forward(input)[0];
        println!(
            "Input: {:?}, Output: {}",
            input
                .iter()
                .map(|v| allocator.get(*v).data)
                .collect::<Vec<_>>(),
            allocator.get(output).data
        );
    }
}
```

## Creating Custom Operators

You can create custom operators by implementing a "forward" and "backward" function for the operator.

```rust
use micrograd_rs::allocator::{Allocator, ValueId};

fn sqrt(x: ValueId<f64>) -> ValueId<f64> {
    unsafe {
        let allocator = x.allocator.as_mut().unwrap();
        let x_val = allocator.get(x).data;
        assert!(
            x_val >= 0.0,
            "Cannot take the square root of a negative number"
        );
        let result = x_val.sqrt();
        allocator.alloc_temp(result, sqrt_backward, [x, ValueId::default()])
    }
}

// base_grad: Gradient of the parent
// base_val: Value of the parent
fn sqrt_backward(
    allocator: &mut Allocator<f64>,
    base_grad: f64,
    base_val: f64,
    children: &[ValueId<f64>],
) {
    // Derivative of sqrt(x) is 0.5 / sqrt(x)
    // In this case, since the operator is basically parent = sqrt(child), we can directly use base_val
    allocator
        .get_mut(children[0])
        .add_grad(base_grad * 0.5 / base_val);
}

fn main() {
    let mut allocator = Allocator::new();
    let a = allocator.alloc(16.0);
    let b = sqrt(a);
    let result = allocator.get(b).data;
    allocator.backward();

    println!("Result: {}, Gradient: {}", result, allocator.get(a).grad);
}
```
