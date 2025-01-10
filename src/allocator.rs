use crate::{engine::Value, operators::Num};

pub type BackwardFn<T> = fn(&mut Allocator<T>, T, T, &[ValueId<T>]);

#[derive(Clone, Copy)]
pub struct ValueId<T: Num> {
    id: i64,
    pub allocator: *mut Allocator<T>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Num> ValueId<T> {
    pub fn step(&self, lr: T) {
        unsafe { (*self.allocator).get_mut(*self).step(lr) }
    }
}

impl<T: Num> Default for ValueId<T> {
    fn default() -> Self {
        Self {
            id: 0,
            allocator: std::ptr::null_mut(),
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct Allocator<T: Num> {
    permanent: Vec<Value<T>>,
    temporary: Vec<Value<T>>,
}

impl<T: Num> Allocator<T> {
    pub fn new() -> Self {
        Self {
            permanent: vec![],
            temporary: vec![],
        }
    }

    pub fn alloc(&mut self, data: T) -> ValueId<T> {
        let id = self.permanent.len();
        self.permanent.push(Value::from(data));
        ValueId {
            id: id as i64,
            allocator: self,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn alloc_t(&mut self, data: T) -> ValueId<T> {
        let id = self.temporary.len() + 1;
        self.temporary.push(Value::from(data));
        ValueId {
            id: -(id as i64),
            allocator: self,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn alloc_temp(
        &mut self,
        data: T,
        backward: BackwardFn<T>,
        previous: [ValueId<T>; 2],
    ) -> ValueId<T> {
        let id = self.temporary.len() + 1;
        self.temporary.push(Value::new(data, backward, previous));
        ValueId {
            id: -(id as i64),
            allocator: self,
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn get(&self, value: ValueId<T>) -> &Value<T> {
        if value.id < 0 {
            &self.temporary[(-value.id - 1) as usize]
        } else {
            &self.permanent[value.id as usize]
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, value: ValueId<T>) -> &mut Value<T> {
        if value.id < 0 {
            &mut self.temporary[(-value.id - 1) as usize]
        } else {
            &mut self.permanent[value.id as usize]
        }
    }

    pub fn zero_grads(&mut self) {
        for value in self.permanent.iter_mut() {
            value.grad = T::zero();
        }

        for value in self.temporary.iter_mut() {
            value.grad = T::zero();
        }
    }

    pub fn clear_temps(&mut self) {
        self.temporary.clear();
    }

    pub fn backward(&mut self) {
        if self.temporary.is_empty() {
            return;
        }

        self.temporary.last_mut().unwrap().grad = T::one();

        for i in (0..self.temporary.len()).rev() {
            let data = self.temporary[i].data;
            let grad = self.temporary[i].grad;
            let previous = self.temporary[i].previous;
            if let Some(backward) = self.temporary[i].backward {
                backward(self, grad, data, &previous);
            }
        }
    }

    pub fn alloc_one_hot(&mut self, index: usize, size: usize, temp: bool) -> Vec<ValueId<T>> {
        let mut ret = Vec::with_capacity(size);
        for i in 0..size {
            if i == index {
                ret.push(if temp {
                    self.alloc_t(T::one())
                } else {
                    self.alloc(T::one())
                });
            } else {
                ret.push(if temp {
                    self.alloc_t(T::zero())
                } else {
                    self.alloc(T::zero())
                });
            }
        }
        ret
    }
}

impl<T: Num> Default for Allocator<T> {
    fn default() -> Self {
        Self::new()
    }
}
