use std::collections::HashSet;

use winit::{
    event::{ElementState, KeyEvent},
    keyboard::{KeyCode, PhysicalKey},
};

pub struct Camera {
    pub position: glam::Vec3,
    front: glam::Vec3,
    up: glam::Vec3,
    right: glam::Vec3,
    world_up: glam::Vec3,
    yaw: f32,
    pitch: f32,
    roll: f32,
    movement_speed: f32,
    mouse_sensitivity: f32,
    fov: f32,
}

const SENSIVITY: f32 = 0.1;
const SPEED: f32 = 20.0;
const FOV: f32 = 90.0;

impl Camera {
    pub fn new(pos: glam::Vec3, up: glam::Vec3, yaw: f32, pitch: f32, roll: f32) -> Self {
        let front = glam::Vec3 {
            x: f32::cos(yaw.to_radians()) * f32::cos(pitch.to_radians()),
            y: f32::sin(pitch.to_radians()),
            z: f32::sin(yaw.to_radians()) * f32::cos(pitch.to_radians()),
        }
        .normalize();
        let right = front.cross(up).normalize();
        Self {
            position: pos,
            front: front,
            up: right.cross(front).normalize(),
            right,
            world_up: up,
            yaw,
            pitch,
            roll,
            movement_speed: SPEED,
            mouse_sensitivity: SENSIVITY,
            fov: FOV,
        }
    }

    pub fn get_view_matrix(&self) -> glam::Mat4 {
        let roll_mat = glam::Mat4::from_axis_angle(self.front, self.roll);
        glam::Mat4::look_at_rh(
            self.position,
            self.position + self.front,
            glam::Mat3::from_mat4(roll_mat) * self.up,
        )
    }

    pub fn process_mouse_movement(&mut self, x: f32, y: f32) {
        self.yaw += x * self.mouse_sensitivity;
        self.pitch += y * self.mouse_sensitivity * -1.0;
        if self.pitch > 89.0 {
            self.pitch = 89.0;
        }
        if self.pitch < -89.0 {
            self.pitch = -89.0;
        }
        self.update_camera_vectors();
    }

    pub fn process_keys(&mut self, pressed_keys: HashSet<KeyCode>, delta_time: f32) {
        let mut movement = glam::Vec3::ZERO;

        for key in pressed_keys {
            match key {
                KeyCode::KeyW => movement += self.front,
                KeyCode::KeyS => movement -= self.front,
                KeyCode::KeyD => movement += self.right,
                KeyCode::KeyA => movement -= self.right,
                KeyCode::Space => movement += self.world_up,
                KeyCode::ShiftLeft => movement -= self.world_up,
                _ => {}
            }
        }

        if movement.length_squared() > 0.0 {
            movement = movement.normalize();
        }

        self.position += movement * self.movement_speed * delta_time;
    }

    fn update_camera_vectors(&mut self) {
        self.front = glam::Vec3 {
            x: f32::cos(self.yaw.to_radians()) * f32::cos(self.pitch.to_radians()),
            y: f32::sin(self.pitch.to_radians()),
            z: f32::sin(self.yaw.to_radians()) * f32::cos(self.pitch.to_radians()),
        }
        .normalize();
        self.right = self.front.cross(self.world_up).normalize();
        self.up = self.right.cross(self.front).normalize()
    }
}
