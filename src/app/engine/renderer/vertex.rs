use anyhow::{Ok, Result};
use ash::vk;
use gpu_allocator::vulkan::Allocator;
use std::{mem, sync::Arc};

use crate::app::engine::renderer::context::{AllocatedBuffer, RenderingContext};

#[repr(C, align(16))]
#[derive(Debug, Clone)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv1: f32,
    pub normal: [f32; 3],
    pub uv2: f32,
}

impl Vertex {
    pub fn new(pos: [f32; 3], normal: [f32; 3], uv1: f32, uv2: f32) -> Self {
        Self {
            pos,
            normal,
            uv1,
            uv2,
        }
    }

    pub fn generate_indices(vertex_count: usize) -> Vec<u32> {
        let indices_count = vertex_count / 4;
        let mut indices = Vec::<u32>::with_capacity(indices_count);
        (0..indices_count).into_iter().for_each(|vert_index| {
            let vert_index = vert_index as u32 * 4u32;
            indices.push(vert_index);
            indices.push(vert_index + 1);
            indices.push(vert_index + 2);
            indices.push(vert_index);
            indices.push(vert_index + 2);
            indices.push(vert_index + 3);
        });
        indices
    }
}
