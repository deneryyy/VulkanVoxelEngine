use std::{collections::HashMap, sync::LazyLock};

use crate::app::game::block::BlockType;

pub struct Registry<T> {
    map: HashMap<u32, T>,
}

pub static BLOCK_REGISTRY: LazyLock<Registry<BlockType>> = LazyLock::new(|| {
    Registry::new(HashMap::from([
        (0, BlockType::new(HashMap::new(), true)),
        (1, BlockType::new(BlockType::side_faces(1, 0), false)),
        (2, BlockType::new(BlockType::all_faces(1), false)),
    ]))
});

impl<T> Registry<T> {
    pub fn new(map: HashMap<u32, T>) -> Self {
        Self { map }
    }

    pub fn get(&self, id: u32) -> Option<&T> {
        self.map.get(&id)
    }

    pub fn register(&mut self, id: u32, value: T) {
        self.map.insert(id, value);
    }
}
