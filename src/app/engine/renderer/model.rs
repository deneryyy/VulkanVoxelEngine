use ash::vk;

use crate::app::engine::renderer::context::AllocatedBuffer;

pub struct Model {
    pub meshes: Vec<Mesh>,
}

pub struct Mesh {
    pub vertex_buffer: AllocatedBuffer,
    pub index_buffer: AllocatedBuffer,
    pub num_elements: u32,
    pub material: usize,
}