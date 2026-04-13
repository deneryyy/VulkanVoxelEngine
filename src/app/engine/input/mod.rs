use std::{
    collections::HashSet,
};

use parking_lot::RwLock;
use winit::keyboard::KeyCode;

pub struct InputManager {
    pub pressed_keys: RwLock<HashSet<KeyCode>>,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            pressed_keys: RwLock::new(HashSet::new()),
        }
    }

    pub fn get_snapshot(&self) -> HashSet<KeyCode> {
        self.pressed_keys.read().clone()
    }
}
