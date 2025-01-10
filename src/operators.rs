use crate::allocator::{Allocator, ValueId};
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
    #[inline(always)]
    fn exp(self) -> Self {
        self.exp()
    }

    #[inline(always)]
    fn ln(self) -> Self {
        self.ln()
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        self.tanh()
    }
}
impl Num for f64 {
    #[inline(always)]
    fn exp(self) -> Self {
        self.exp()
    }

    #[inline(always)]
    fn ln(self) -> Self {
        self.ln()
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        self.tanh()
    }
}

impl<T: Num> Add for ValueId<T> {
    type Output = ValueId<T>;

    #[inline(always)]
    fn add(self, other: ValueId<T>) -> ValueId<T> {
        assert!(self.allocator == other.allocator);

        unsafe {
            let allocator = self.allocator.as_mut().unwrap();
            let result = allocator.get(self).data + allocator.get(other).data;
            allocator.alloc_temp(result, add_backward::<T>, [self, other])
        }
    }
}

fn add_backward<T: Num>(
    allocator: &mut Allocator<T>,
    base_grad: T,
    _base_val: T,
    children: &[ValueId<T>],
) {
    allocator.get_mut(children[0]).add_grad(base_grad);
    allocator.get_mut(children[1]).add_grad(base_grad);
}

impl<T: Num + Copy> Mul for ValueId<T> {
    type Output = ValueId<T>;

    #[inline(always)]
    fn mul(self, other: ValueId<T>) -> ValueId<T> {
        assert!(self.allocator == other.allocator);

        unsafe {
            let allocator = self.allocator.as_mut().unwrap();
            let result = allocator.get(self).data * allocator.get(other).data;
            allocator.alloc_temp(result, mul_backward::<T>, [self, other])
        }
    }
}

fn mul_backward<T: Num>(
    allocator: &mut Allocator<T>,
    base_grad: T,
    _base_val: T,
    children: &[ValueId<T>],
) {
    let a = allocator.get(children[0]).data;
    let b = allocator.get(children[1]).data;
    allocator.get_mut(children[0]).add_grad(base_grad * b);
    allocator.get_mut(children[1]).add_grad(base_grad * a);
}

impl<T: Num> Neg for ValueId<T> {
    type Output = ValueId<T>;

    #[inline(always)]
    fn neg(self) -> ValueId<T> {
        unsafe {
            let allocator = self.allocator.as_mut().unwrap();
            let result = allocator.get(self).data * -T::one();
            allocator.alloc_temp(result, neg_backward::<T>, [self, ValueId::default()])
        }
    }
}

fn neg_backward<T: Num>(
    allocator: &mut Allocator<T>,
    base_grad: T,
    _base_val: T,
    children: &[ValueId<T>],
) {
    allocator.get_mut(children[0]).add_grad(-base_grad);
}

impl<T: Num> Sub for ValueId<T> {
    type Output = ValueId<T>;

    #[inline(always)]
    fn sub(self, other: ValueId<T>) -> ValueId<T> {
        self + -other
    }
}

#[inline(always)]
pub fn pow<T: Num>(this: ValueId<T>, other: ValueId<T>) -> ValueId<T> {
    assert!(this.allocator == other.allocator);

    unsafe {
        let allocator = this.allocator.as_mut().unwrap();
        let result = allocator.get(this).data.pow(allocator.get(other).data);
        allocator.alloc_temp(result, pow_backward::<T>, [this, other])
    }
}

fn pow_backward<T: Num>(
    allocator: &mut Allocator<T>,
    base_grad: T,
    base_val: T,
    children: &[ValueId<T>],
) {
    let a = allocator.get(children[0]).data;
    let b = allocator.get(children[1]).data;
    allocator
        .get_mut(children[0])
        .add_grad(base_grad * b * base_val / a);
    allocator
        .get_mut(children[1])
        .add_grad(base_grad * base_val * a.ln());
}

#[inline(always)]
pub fn exp<T: Num>(this: ValueId<T>) -> ValueId<T> {
    unsafe {
        let allocator = this.allocator.as_mut().unwrap();
        let result = allocator.get(this).data.exp();
        allocator.alloc_temp(result, exp_backward::<T>, [this, ValueId::default()])
    }
}

fn exp_backward<T: Num>(
    allocator: &mut Allocator<T>,
    base_grad: T,
    base_val: T,
    children: &[ValueId<T>],
) {
    allocator
        .get_mut(children[0])
        .add_grad(base_grad * base_val);
}

#[inline(always)]
pub fn ln<T: Num>(v: ValueId<T>) -> ValueId<T> {
    unsafe {
        let allocator = v.allocator.as_mut().unwrap();
        let result = allocator.get(v).data.ln();
        allocator.alloc_temp(result, ln_backward::<T>, [v, ValueId::default()])
    }
}

fn ln_backward<T: Num>(
    allocator: &mut Allocator<T>,
    base_grad: T,
    _base_val: T,
    children: &[ValueId<T>],
) {
    let a = allocator.get(children[0]).data;
    allocator
        .get_mut(children[0])
        .add_grad(base_grad * T::one() / a);
}

#[inline(always)]
pub fn tanh<T: Num>(this: ValueId<T>) -> ValueId<T> {
    unsafe {
        let allocator = this.allocator.as_mut().unwrap();
        let result = allocator.get(this).data.tanh();
        allocator.alloc_temp(result, tanh_backward::<T>, [this, ValueId::default()])
    }
}

fn tanh_backward<T: Num>(
    allocator: &mut Allocator<T>,
    base_grad: T,
    base_val: T,
    children: &[ValueId<T>],
) {
    allocator
        .get_mut(children[0])
        .add_grad(base_grad * (T::one() - base_val.pow(T::one() + T::one())));
}

#[inline(always)]
pub fn relu<T: Num>(this: ValueId<T>) -> ValueId<T> {
    unsafe {
        let allocator = this.allocator.as_mut().unwrap();
        let result = if allocator.get(this).data > T::zero() {
            allocator.get(this).data
        } else {
            T::zero()
        };
        allocator.alloc_temp(result, relu_backward::<T>, [this, ValueId::default()])
    }
}

fn relu_backward<T: Num>(
    allocator: &mut Allocator<T>,
    base_grad: T,
    base_val: T,
    children: &[ValueId<T>],
) {
    allocator.get_mut(children[0]).add_grad(
        base_grad
            * if base_val > T::zero() {
                T::one()
            } else {
                T::zero()
            },
    );
}

impl<T: Num> Div for ValueId<T> {
    type Output = ValueId<T>;

    #[inline(always)]
    fn div(self, other: ValueId<T>) -> ValueId<T> {
        assert!(self.allocator == other.allocator);

        unsafe {
            let allocator = self.allocator.as_mut().unwrap();
            let result = allocator.get(self).data / allocator.get(other).data;
            allocator.alloc_temp(result, div_backward::<T>, [self, other])
        }
    }
}

fn div_backward<T: Num>(
    allocator: &mut Allocator<T>,
    base_grad: T,
    base_val: T,
    children: &[ValueId<T>],
) {
    let b = allocator.get(children[1]).data;

    allocator
        .get_mut(children[0])
        .add_grad(base_grad * T::one() / b);
    allocator
        .get_mut(children[1])
        .add_grad(base_grad * -base_val * T::one() / b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let mut allocator = Allocator::new();
        let a = allocator.alloc(3.0);
        let b = allocator.alloc(4.0);
        let c = a + b;
        assert_eq!(allocator.get(c).data, 7.0);

        allocator.backward();
        assert_eq!(allocator.get(a).grad, 1.0);
        assert_eq!(allocator.get(b).grad, 1.0);
        assert_eq!(allocator.get(c).grad, 1.0);
    }

    #[test]
    fn test_multiplication() {
        let mut allocator = Allocator::new();
        let a = allocator.alloc(3.0);
        let b = allocator.alloc(4.0);
        let c = a * b;
        assert_eq!(allocator.get(c).data, 12.0);

        allocator.backward();
        assert_eq!(allocator.get(a).grad, 4.0);
        assert_eq!(allocator.get(b).grad, 3.0);
        assert_eq!(allocator.get(c).grad, 1.0);
    }

    #[test]
    fn test_negation() {
        let mut allocator = Allocator::new();
        let a = allocator.alloc(3.0);
        let b = -a;
        assert_eq!(allocator.get(b).data, -3.0);

        allocator.backward();
        assert_eq!(allocator.get(a).grad, -1.0);
        assert_eq!(allocator.get(b).grad, 1.0);
    }

    #[test]
    fn test_subtraction() {
        let mut allocator = Allocator::new();
        let a = allocator.alloc(3.0);
        let b = allocator.alloc(4.0);
        let c = a - b;
        assert_eq!(allocator.get(c).data, -1.0);

        allocator.backward();
        assert_eq!(allocator.get(a).grad, 1.0);
        assert_eq!(allocator.get(b).grad, -1.0);
        assert_eq!(allocator.get(c).grad, 1.0);
    }

    #[test]
    fn test_pow() {
        let mut allocator = Allocator::new();
        let a = allocator.alloc(3.0);
        let b = allocator.alloc(4.0);
        let c = pow(a, b);
        assert_eq!(allocator.get(c).data, 81.0);

        allocator.backward();
        assert_eq!(allocator.get(a).grad, 108.0);
        assert_eq!(allocator.get(b).grad, 88.9875953821169);
        assert_eq!(allocator.get(c).grad, 1.0);
    }

    #[test]
    fn test_exp() {
        let mut allocator = Allocator::new();
        let a = allocator.alloc(3.0);
        let b = exp(a);
        assert_eq!(allocator.get(b).data, 20.085536923187668);

        allocator.backward();
        assert_eq!(allocator.get(a).grad, 20.085536923187668);
        assert_eq!(allocator.get(b).grad, 1.0);
    }

    #[test]
    fn test_division() {
        let mut allocator = Allocator::new();
        let a = allocator.alloc(3.0);
        let b = allocator.alloc(4.0);
        let c = a / b;
        assert_eq!(allocator.get(c).data, 0.75);

        allocator.backward();
        assert_eq!(allocator.get(a).grad, 0.25);
        assert_eq!(allocator.get(b).grad, -0.1875);
        assert_eq!(allocator.get(c).grad, 1.0);
    }

    #[test]
    fn test_tanh() {
        let mut allocator = Allocator::new();
        let a = allocator.alloc(3.0);
        let b = tanh(a);
        assert_eq!(allocator.get(b).data, 0.9950547536867305);

        allocator.backward();
        assert_eq!(allocator.get(a).grad, 0.009866037165440211);
        assert_eq!(allocator.get(b).grad, 1.0);
    }

    #[test]
    fn test_relu() {
        let mut allocator = Allocator::new();
        let a = allocator.alloc(3.0);
        let b = relu(a);
        assert_eq!(allocator.get(b).data, 3.0);

        allocator.backward();
        assert_eq!(allocator.get(a).grad, 1.0);
        assert_eq!(allocator.get(b).grad, 1.0);

        let c = allocator.alloc(-3.0);
        let d = relu(c);
        assert_eq!(allocator.get(d).data, 0.0);

        allocator.backward();
        assert_eq!(allocator.get(c).grad, 0.0);
        assert_eq!(allocator.get(d).grad, 1.0);
    }
}
