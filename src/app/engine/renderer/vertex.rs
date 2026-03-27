use ash::vk;
use std::mem;

#[repr(C, align(16))]
pub struct Vertex {
    pub pos: [f32; 3],
    pub pad1: f32,
    pub color: [f32; 3],
    pub pad2: f32
}

impl Vertex {
    
}
