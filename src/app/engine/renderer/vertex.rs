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

    pub fn create_vertex_and_index_buffers(
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
    ) -> Result<(AllocatedBuffer, AllocatedBuffer, vk::DeviceAddress)> {
        unsafe {
            let vertex_buffer_size = (vertices.len() * std::mem::size_of::<Vertex>()) as u64;
            let vertex_buffer = context.create_buffer(
                allocator,
                vertex_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )?;

            let index_buffer_size = (indices.len() * std::mem::size_of::<u32>()) as u64;
            let index_buffer = context.create_buffer(
                allocator,
                index_buffer_size,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )?;
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr(),
                vertex_buffer
                    .allocation
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr()
                    .cast(),
                vertices.len(),
            );
            std::ptr::copy_nonoverlapping(
                indices.as_ptr(),
                index_buffer
                    .allocation
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr()
                    .cast(),
                indices.len(),
            );

            let vertex_buffer_address = context.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer),
            );

            Ok((vertex_buffer, index_buffer, vertex_buffer_address))
        }
    }
}
