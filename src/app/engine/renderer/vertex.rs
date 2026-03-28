use ash::vk;
use std::mem;

#[repr(C, align(16))]
#[derive(Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub pad1: f32,
    pub normal: [f32; 3],
    pub pad2: f32,
    pub color: [f32; 3],
    pub pad3: f32,
}

impl Vertex {
   pub fn new(pos: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
    Self { pos, pad1: 0.0, normal, pad2: 0.0, color, pad3: 0.0 }
   } 
}
