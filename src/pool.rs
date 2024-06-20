use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use bumpalo::Bump;
use flume::{Receiver, Sender, TrySendError};
use once_cell::sync::Lazy;
use parking_lot::Mutex;

pub static BUMP: Lazy<SimplePool<Bump>> =
    Lazy::new(|| SimplePool::new(|| Bump::with_capacity(1024 * 1024)));

pub trait Reset: Sized {
    fn reset(self) -> Option<Self>;
}

impl Reset for Bump {
    fn reset(mut self) -> Option<Self> {
        Bump::reset(&mut self);
        Some(self)
    }
}

impl<T> Reset for Vec<T> {
    fn reset(mut self) -> Option<Self> {
        self.clear();
        Some(self)
    }
}

impl<K, V> Reset for HashMap<K, V> {
    fn reset(mut self) -> Option<Self> {
        self.clear();
        Some(self)
    }
}

impl<K, V> Reset for bevy::utils::HashMap<K, V> {
    fn reset(mut self) -> Option<Self> {
        self.clear();
        Some(self)
    }
}

struct PoolItem<T> {
    sender: Sender<T>,
    receiver: Mutex<Receiver<T>>,
}

pub struct PoolLease<'a, T: Reset>(Option<T>, &'a SimplePool<T>);

impl<'a, T: Reset> Drop for PoolLease<'a, T> {
    fn drop(&mut self) {
        if let Some(mut v) = self.0.take().and_then(Reset::reset) {
            for item in self.1 .0.iter() {
                match item.sender.try_send(v) {
                    Ok(_) => return,
                    Err(TrySendError::Full(r)) => v = r,
                    _ => return,
                }
            }
        }
    }
}

impl<'a, T: Reset> Deref for PoolLease<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}

impl<'a, T: Reset> DerefMut for PoolLease<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().unwrap()
    }
}

impl<'a, T: Reset> AsRef<T> for PoolLease<'a, T> {
    fn as_ref(&self) -> &T {
        self.deref()
    }
}

impl<'a, T: Reset> AsMut<T> for PoolLease<'a, T> {
    fn as_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

pub struct SimplePool<T: Reset>([PoolItem<T>; 8], Box<dyn Send + Sync + Fn() -> T>);

impl<T: 'static + Reset + Default> Default for SimplePool<T> {
    fn default() -> Self {
        Self::new(Default::default)
    }
}

impl<T: 'static + Reset + Default> SimplePool<T> {
    pub fn default_with_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity, Default::default)
    }
}

impl<T: Reset> SimplePool<T> {
    pub fn new<F: 'static + Send + Sync + Fn() -> T>(f: F) -> Self {
        Self::with_capacity(128, f)
    }

    pub fn with_capacity<F: 'static + Send + Sync + Fn() -> T>(capacity: usize, f: F) -> Self {
        Self(
            std::array::from_fn(|_| {
                let (sender, receiver) = flume::bounded(capacity / 8);
                let receiver = Mutex::new(receiver);
                PoolItem { sender, receiver }
            }),
            Box::new(f),
        )
    }

    pub fn take(&self) -> PoolLease<'_, T> {
        for item in self.0.iter() {
            if let Some(v) = item.receiver.try_lock() {
                if let Ok(v) = v.try_recv() {
                    return PoolLease(Some(v), self);
                }
            }
        }
        PoolLease(Some(self.1()), self)
    }
}
