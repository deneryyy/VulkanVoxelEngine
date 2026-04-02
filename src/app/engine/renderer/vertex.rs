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
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;

            let index_buffer_size = (indices.len() * std::mem::size_of::<u32>()) as u64;
            let index_buffer = context.create_buffer(
                allocator,
                index_buffer_size,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;

            let mut staging = context.create_buffer(
                allocator,
                vertex_buffer_size + index_buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                gpu_allocator::MemoryLocation::CpuToGpu,
            )?;

            let data = staging.allocation.mapped_ptr().unwrap().as_ptr();
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), data.cast(), vertices.len());
            std::ptr::copy_nonoverlapping(
                indices.as_ptr(),
                data.cast::<u32>().byte_add(vertex_buffer_size as usize),
                indices.len(),
            );

            let cmd = context.begin_immediate_submit()?;

            context.device.cmd_copy_buffer(
                cmd,
                staging.buffer,
                vertex_buffer.buffer,
                &[vk::BufferCopy::default()
                    .dst_offset(0)
                    .src_offset(0)
                    .size(vertex_buffer_size)],
            );
            context.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(vertex_buffer.buffer)
                        .size(vertex_buffer_size),
                ]),
            );
            context.device.cmd_copy_buffer(
                cmd,
                staging.buffer,
                index_buffer.buffer,
                &[vk::BufferCopy::default()
                    .dst_offset(0)
                    .src_offset(vertex_buffer_size)
                    .size(index_buffer_size)],
            );
            context.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::INDEX_INPUT)
                        .dst_access_mask(vk::AccessFlags2::INDEX_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(index_buffer.buffer)
                        .size(index_buffer_size),
                ]),
            );

            context.end_immediate_submit()?;
            let vertex_buffer_address = context.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer),
            );
            context.destroy_buffer(allocator, &mut staging);
            Ok((vertex_buffer, index_buffer, vertex_buffer_address))
        }
    }
}
