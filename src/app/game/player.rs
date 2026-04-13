use std::{collections::HashSet, sync::Arc};

use glam::Vec3;
use winit::keyboard::KeyCode;

use crate::app::engine::{input::InputManager, renderer::camera::Camera};

pub struct Player {
    input_manager: Arc<InputManager>,
    pub position: Vec3,
    pub camera: Camera,
    pub render_distance: u32,
}

impl Player {
    pub fn new(input_manager: Arc<InputManager>, position: Vec3, camera: Camera, render_distance: u32) -> Self {
        Self {
            input_manager,
            position,
            camera,
            render_distance,
        }
    }

    pub fn tick(&mut self, delta_time: f32) {
        let pressed_keys: HashSet<KeyCode> = self.input_manager.get_snapshot();
        self.camera.process_keys(pressed_keys, delta_time);
        self.position = self.camera.position;
    }
}
