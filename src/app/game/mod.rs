use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::app::game::{world::World};

pub mod block;
pub mod registry;
pub mod world;

pub struct Game {
    world: Arc<Mutex<World>>,
}

impl Game {
    pub fn new(world: Arc<Mutex<World>>) -> Self {
        Self {
            world,
        }
    }

    pub fn start_game(&mut self) {
        {
            self.world.lock().unwrap().generate();
        }
        let mut last_frame = Instant::now();
        loop {
            let now = Instant::now();
            let delta_time = now.duration_since(last_frame).as_secs_f32();
            last_frame = now;
        }
    }
}
