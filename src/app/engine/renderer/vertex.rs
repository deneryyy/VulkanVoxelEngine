use ash::vk;
use std::mem;

#[repr(C, align(16))]
#[derive(Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv1: f32,
    pub normal: [f32; 3],
    pub uv2: f32,
    pub color: [f32; 3],
    pub pad3: f32,
}

impl Vertex {
    pub fn new(pos: [f32; 3], normal: [f32; 3], color: [f32; 3], uv1: f32, uv2: f32) -> Self {
        Self {
            pos,
            normal,
            color,
            uv1,
            uv2,
            pad3: 0.0,
        }
    }
}
