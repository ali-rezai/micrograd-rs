use crate::engine::Value;
use num::pow::Pow;
use num::Num as BaseNum;
use rand::distributions::uniform::SampleUniform;
use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
};

pub trait Num:
    BaseNum
    + Neg<Output = Self>
    + Pow<Self, Output = Self>
    + Copy
    + Display
    + PartialOrd
    + SampleUniform
{
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn tanh(self) -> Self;
}
impl Num for f32 {
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn tanh(self) -> Self {
        self.tanh()
    }
}
impl Num for f64 {
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn tanh(self) -> Self {
        self.tanh()
    }
}

impl<T: Num> Add for Value<T> {
    type Output = Value<T>;

    fn add(self, other: Value<T>) -> Value<T> {
        let result = self.data() + other.data();
        Value::new(result, add_backward, vec![self, other])
    }
}

fn add_backward<T: Num>(base_grad: T, _base_val: T, children: &mut [Value<T>]) {
    children[0].add_grad(base_grad);
    children[1].add_grad(base_grad);
}

impl<T: Num + Copy> Mul for Value<T> {
    type Output = Value<T>;

    fn mul(self, other: Value<T>) -> Value<T> {
        let result = self.data() * other.data();
        Value::new(result, mul_backward, vec![self, other])
    }
}

fn mul_backward<T: Num>(base_grad: T, _base_val: T, children: &mut [Value<T>]) {
    let a = children[0].data();
    let b = children[1].data();
    children[0].add_grad(base_grad * b);
    children[1].add_grad(base_grad * a);
}

impl<T: Num> Neg for Value<T> {
    type Output = Value<T>;

    fn neg(self) -> Value<T> {
        self * Value::from(-T::one())
    }
}

impl<T: Num> Sub for Value<T> {
    type Output = Value<T>;

    fn sub(self, other: Value<T>) -> Value<T> {
        self + -other
    }
}

pub fn pow<T: Num>(this: Value<T>, other: Value<T>) -> Value<T> {
    let result = this.data().pow(other.data());
    Value::new(result, pow_backward, vec![this, other])
}

fn pow_backward<T: Num>(base_grad: T, base_val: T, children: &mut [Value<T>]) {
    let a = children[0].data();
    let b = children[1].data();
    children[0].add_grad(base_grad * b * base_val / a);
    children[1].add_grad(base_grad * base_val * a.ln());
}

pub fn exp<T: Num>(this: Value<T>) -> Value<T> {
    let result = this.data().exp();
    Value::new(result, exp_backward, vec![this])
}

fn exp_backward<T: Num>(base_grad: T, base_val: T, children: &mut [Value<T>]) {
    children[0].add_grad(base_grad * base_val);
}

pub fn tanh<T: Num>(this: Value<T>) -> Value<T> {
    let result = this.data().tanh();
    Value::new(result, tanh_backward, vec![this])
}

fn tanh_backward<T: Num>(base_grad: T, base_val: T, children: &mut [Value<T>]) {
    children[0].add_grad(base_grad * (T::one() - base_val.pow(T::one() + T::one())));
}

pub fn relu<T: Num>(this: Value<T>) -> Value<T> {
    let result = if this.data() > T::zero() {
        this.data()
    } else {
        T::zero()
    };
    Value::new(result, relu_backward, vec![this])
}

fn relu_backward<T: Num>(base_grad: T, base_val: T, children: &mut [Value<T>]) {
    children[0].add_grad(
        base_grad
            * if base_val > T::zero() {
                T::one()
            } else {
                T::zero()
            },
    );
}

impl<T: Num> Div for Value<T> {
    type Output = Value<T>;

    fn div(self, other: Value<T>) -> Value<T> {
        self * pow(other, Value::from(-T::one()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let a = Value::from(3.0);
        let b = Value::from(4.0);
        let c = a.clone() + b.clone();
        assert_eq!(c.data(), 7.0);

        c.backward();
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn test_multiplication() {
        let a = Value::from(3.0);
        let b = Value::from(4.0);
        let c = a.clone() * b.clone();
        assert_eq!(c.data(), 12.0);

        c.backward();
        assert_eq!(a.grad(), 4.0);
        assert_eq!(b.grad(), 3.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn test_negation() {
        let a = Value::from(3.0);
        let b = -a.clone();
        assert_eq!(b.data(), -3.0);

        b.backward();
        assert_eq!(a.grad(), -1.0);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn test_subtraction() {
        let a = Value::from(3.0);
        let b = Value::from(4.0);
        let c = a.clone() - b.clone();
        assert_eq!(c.data(), -1.0);

        c.backward();
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), -1.0);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn test_pow() {
        let a = Value::from(3.0);
        let b = Value::from(4.0);
        let c = pow(a.clone(), b.clone());
        assert_eq!(c.data(), 81.0);

        c.backward();
        assert_eq!(a.grad(), 108.0);
        assert_eq!(b.grad(), 88.9875953821169);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn test_exp() {
        let a = Value::from(3.0);
        let b = exp(a.clone());
        assert_eq!(b.data(), 20.085536923187668);

        b.backward();
        assert_eq!(a.grad(), 20.085536923187668);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn test_division() {
        let a = Value::from(3.0);
        let b = Value::from(4.0);
        let c = a.clone() / b.clone();
        assert_eq!(c.data(), 0.75);

        c.backward();
        assert_eq!(a.grad(), 0.25);
        assert_eq!(b.grad(), -0.1875);
        assert_eq!(c.grad(), 1.0);
    }

    #[test]
    fn test_tanh() {
        let a = Value::from(3.0);
        let b = tanh(a.clone());
        assert_eq!(b.data(), 0.9950547536867305);

        b.backward();
        assert_eq!(a.grad(), 0.009866037165440211);
        assert_eq!(b.grad(), 1.0);
    }

    #[test]
    fn test_relu() {
        let a = Value::from(3.0);
        let b = relu(a.clone());
        assert_eq!(b.data(), 3.0);

        b.backward();
        assert_eq!(a.grad(), 1.0);
        assert_eq!(b.grad(), 1.0);

        let c = Value::from(-3.0);
        let d = relu(c.clone());
        assert_eq!(d.data(), 0.0);

        d.backward();
        assert_eq!(c.grad(), 0.0);
        assert_eq!(d.grad(), 1.0);
    }
}
