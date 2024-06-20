use once_cell::sync::Lazy;

use crate::pool::{PoolLease, SimplePool};

///! gpu ready mesh payload
static VEC: Lazy<SimplePool<Vec<u32>>> = Lazy::new(SimplePool::default);

pub struct ChunkMesh {
    pub indices: PoolLease<'static, Vec<u32>>,
    pub vertices: PoolLease<'static, Vec<u32>>,
}

impl Default for ChunkMesh {
    fn default() -> Self {
        Self {
            indices: VEC.take(),
            vertices: VEC.take(),
        }
    }
}
