use winit::{
    event::{ElementState, KeyEvent},
    keyboard::{KeyCode, PhysicalKey},
};

pub struct Camera {
    position: glam::Vec3,
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
const SPEED: f32 = 0.5;
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

    pub fn process_key_movement(&mut self, key: KeyCode, state: ElementState) {
        if key == KeyCode::KeyW && state.is_pressed() {
            self.position += self.front * self.movement_speed;
        } else if key == KeyCode::KeyS && state.is_pressed() {
            self.position -= self.front * self.movement_speed;
        } else if key == KeyCode::KeyA && state.is_pressed() {
            self.position -= self.right * self.movement_speed;
        } else if key == KeyCode::KeyD && state.is_pressed() {
            self.position += self.right * self.movement_speed
        } else if key == KeyCode::Space && state.is_pressed() {
            self.position += self.world_up * self.movement_speed
        } else if key == KeyCode::ShiftLeft && state.is_pressed() {
            self.position -= self.world_up * self.movement_speed
        }
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
