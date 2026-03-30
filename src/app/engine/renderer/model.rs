use ash::vk;

use crate::app::engine::renderer::context::AllocatedBuffer;

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

pub struct Material {
    pub name: String,
    //pub diffuse_texture: texture::Texture,
    //pub bind_group: wgpu::BindGroup,
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: AllocatedBuffer,
    pub index_buffer: AllocatedBuffer,
    pub num_elements: u32,
    pub material: usize,
}