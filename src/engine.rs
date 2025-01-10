use crate::allocator::{BackwardFn, ValueId};
use crate::operators::Num;
use std::fmt::Debug;

#[derive(Clone)]
pub struct Value<T: Num> {
    pub data: T,
    pub grad: T,
    pub(crate) previous: [ValueId<T>; 2],
    pub(crate) backward: Option<BackwardFn<T>>,
}

impl Debug for Value<f32> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Value").field(&self.data).finish()
    }
}

impl<T: Num> Value<T> {
    pub fn from(data: T) -> Value<T> {
        Value {
            data,
            grad: T::zero(),
            backward: None,
            previous: [ValueId::default(), ValueId::default()],
        }
    }

    pub fn new(data: T, backward: BackwardFn<T>, previous: [ValueId<T>; 2]) -> Value<T> {
        Value {
            data,
            grad: T::zero(),
            backward: Some(backward),
            previous,
        }
    }

    pub fn set_data(&mut self, data: T) {
        self.data = data;
    }

    #[inline(always)]
    pub fn step(&mut self, lr: T) {
        self.data = self.data - lr * self.grad;
        self.grad = T::zero();
    }

    #[inline(always)]
    pub fn add_grad(&mut self, grad: T) {
        self.grad = self.grad + grad;
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        allocator::Allocator,
        operators::{exp, pow},
    };

    #[test]
    fn test_value_creation() {
        let mut allocator = Allocator::new();
        let value = allocator.alloc(3.0);
        assert_eq!(allocator.get(value).data, 3.0);
    }

    #[test]
    fn test_backward_and_rezero() {
        let mut allocator = Allocator::new();
        let a = allocator.alloc(3.0);
        let b = allocator.alloc(2.0);
        let c = pow(a, b);
        let d = c + c;
        let e = d * a;
        let f = e - d;
        let g = f / c;
        let h = exp(g);
        assert_eq!(allocator.get(h).data, 54.598150033144236);

        allocator.backward();
        assert_eq!(allocator.get(a).grad, 109.1963000662885);
        assert_eq!(allocator.get(b).grad, 3.512749415084566e-14);
        assert_eq!(allocator.get(c).grad, 3.552713678800501e-15);
        assert_eq!(allocator.get(d).grad, 12.13292222958761);
        assert_eq!(allocator.get(e).grad, 6.066461114793804);
        assert_eq!(allocator.get(f).grad, 6.066461114793804);
        assert_eq!(allocator.get(g).grad, 54.598150033144236);
        assert_eq!(allocator.get(h).grad, 1.0);

        allocator.zero_grads();
        assert_eq!(allocator.get(a).grad, 0.0);
        assert_eq!(allocator.get(b).grad, 0.0);
        assert_eq!(allocator.get(c).grad, 0.0);
        assert_eq!(allocator.get(d).grad, 0.0);
        assert_eq!(allocator.get(e).grad, 0.0);
        assert_eq!(allocator.get(f).grad, 0.0);
        assert_eq!(allocator.get(g).grad, 0.0);
        assert_eq!(allocator.get(h).grad, 0.0);
    }
}
