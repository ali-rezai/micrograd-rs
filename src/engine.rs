use crate::operators::Num;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Debug;
use std::rc::Rc;

type BackwardFn<T> = fn(T, T, &mut [Value<T>]);

#[derive(Clone)]
pub struct Value<T: Num>(Rc<RefCell<ValueInner<T>>>);

impl Debug for Value<f32> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Value").field(&self.data()).finish()
    }
}

impl<T: Num> Value<T> {
    pub fn from(data: T) -> Value<T> {
        Value(Rc::new(RefCell::new(ValueInner::new(
            data,
            None,
            Vec::new(),
        ))))
    }

    pub fn new(data: T, backward: BackwardFn<T>, previous: Vec<Value<T>>) -> Value<T> {
        Value(Rc::new(RefCell::new(ValueInner::new(
            data,
            Some(backward),
            previous,
        ))))
    }

    pub fn data(&self) -> T {
        self.0.borrow().data
    }

    pub fn step(&self, lr: T) {
        let mut inner = self.0.borrow_mut();
        inner.data = inner.data - lr * inner.gradient;
    }

    pub fn add_grad(&self, grad: T) {
        let mut inner = self.0.borrow_mut();
        inner.gradient = inner.gradient + grad;
    }

    pub fn grad(&self) -> T {
        self.0.borrow().gradient
    }

    pub fn backward(&self) {
        let mut inner = self.0.borrow_mut();
        inner.gradient = T::one();
        drop(inner);

        let mut visited = HashSet::new();
        let mut topo = Vec::new();
        fn dfs<T: Num>(
            node: Value<T>,
            visited: &mut HashSet<*mut ValueInner<T>>,
            topo: &mut Vec<Value<T>>,
        ) {
            if !visited.insert(node.0.as_ptr()) {
                return;
            }
            let inner = node.0.borrow();
            for child in inner.previous.iter() {
                dfs(child.clone(), visited, topo);
            }
            drop(inner);
            topo.push(node);
        }
        dfs(self.clone(), &mut visited, &mut topo);

        for node in topo.iter().rev() {
            let mut inner = node.0.borrow_mut();
            if inner.backward.is_none() {
                continue;
            }
            inner.backward.unwrap()(inner.gradient, inner.data, &mut inner.previous);
        }
    }

    pub fn zero_grads(&self) {
        let mut inner = self.0.borrow_mut();
        inner.gradient = T::zero();
        drop(inner);

        let mut visited = HashSet::new();
        visited.insert(self.0.as_ptr());
        let mut stack = self.0.borrow().previous.clone();
        while let Some(current) = stack.pop() {
            if !visited.insert(current.0.as_ptr()) {
                continue;
            }

            let mut inner = current.0.borrow_mut();
            inner.gradient = T::zero();
            drop(inner);

            stack.extend_from_slice(&current.0.borrow().previous);
        }
    }
}

struct ValueInner<T: Num> {
    data: T,
    gradient: T,
    backward: Option<BackwardFn<T>>,
    previous: Vec<Value<T>>,
}

impl<T: Num> ValueInner<T> {
    fn new(data: T, backward: Option<BackwardFn<T>>, previous: Vec<Value<T>>) -> ValueInner<T> {
        ValueInner {
            data,
            gradient: T::zero(),
            backward,
            previous,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::operators::{exp, pow};

    use super::*;

    #[test]
    fn test_value_creation() {
        let value = Value::from(3.0);
        assert_eq!(value.data(), 3.0);
    }

    #[test]
    fn test_backward_and_rezero() {
        let a = Value::from(3.0);
        let b = Value::from(2.0);
        let c = pow(a.clone(), b.clone());
        let d = c.clone() + c.clone();
        let e = d.clone() * a.clone();
        let f = e.clone() - d.clone();
        let g = f.clone() / c.clone();
        let h = exp(g.clone());
        assert_eq!(h.data(), 54.598150033144236);

        h.backward();
        assert_eq!(a.grad(), 109.1963000662885);
        assert_eq!(b.grad(), 3.512749415084566e-14);
        assert_eq!(c.grad(), 3.552713678800501e-15);
        assert_eq!(d.grad(), 12.13292222958761);
        assert_eq!(e.grad(), 6.066461114793804);
        assert_eq!(f.grad(), 6.066461114793804);
        assert_eq!(g.grad(), 54.598150033144236);
        assert_eq!(h.grad(), 1.0);

        h.zero_grads();
        assert_eq!(a.grad(), 0.0);
        assert_eq!(b.grad(), 0.0);
        assert_eq!(c.grad(), 0.0);
        assert_eq!(d.grad(), 0.0);
        assert_eq!(e.grad(), 0.0);
        assert_eq!(f.grad(), 0.0);
        assert_eq!(g.grad(), 0.0);
        assert_eq!(h.grad(), 0.0);
    }
}
