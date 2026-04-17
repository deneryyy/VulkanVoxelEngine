use std::{collections::VecDeque, sync::Arc};

use anyhow::{Ok, Result, anyhow};
use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::app::engine::renderer::{
    FRAME_OVERLAP,
    context::{AllocatedBuffer, RenderingContext},
};

pub struct RingStagingBuffer {
    pub buffer: AllocatedBuffer,
    size: u64,
    head: u64,
    frame_allocations: Vec<Vec<RingStagingBufferAllocation>>,
}

#[derive(Clone, Copy)]
pub struct RingStagingBufferAllocation {
    size: u64,
    offset: u64,
}

impl RingStagingBuffer {
    pub fn new(
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
        size: u64,
    ) -> Result<Self> {
        let buffer = context.create_buffer(
            "Ring staging buffer",
            allocator,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
        )?;
        Ok(Self {
            buffer,
            size,
            head: 0,
            frame_allocations: (0..FRAME_OVERLAP).map(|_| Vec::new()).collect(),
        })
    }

    pub fn can_allocate<T>(&self, data: &[T], frame_index: usize) -> Result<()> {
        let size = (std::mem::size_of::<T>() * data.len()) as u64;
        let alignment = std::mem::align_of::<T>() as u64;

        let mut offset = (self.head + alignment - 1) & !(alignment - 1);

        let wraps = offset + size > self.size;

        if wraps {
            offset = 0;
        }

        for (index, allocations) in self.frame_allocations.iter().enumerate() {
            if index != frame_index {
                for existing in allocations {
                    let overlap =
                        offset < existing.offset + existing.size && offset + size > existing.offset;
                    if overlap {
                        return Err(anyhow!(
                            "Buffer full: would overwrite frame {} alloc at offset {}-{}",
                            index,
                            existing.offset,
                            existing.offset + existing.size
                        ));
                    }
                }
            }
        }

        for existing in &self.frame_allocations[frame_index] {
            let overlap =
                offset < existing.offset + existing.size && offset + size > existing.offset;
            if overlap {
                return Err(anyhow!("Overlap with same frame's existing allocation"));
            }
        }

        Ok(())
    }

    pub fn push<T>(
        &mut self,
        data: &[T],
        frame_index: usize,
    ) -> Result<RingStagingBufferAllocation> {
        self.can_allocate(data, frame_index)?;
        let size = (std::mem::size_of::<T>() * data.len()) as u64;
        let alignment = std::mem::align_of::<T>() as u64;

        let mut offset = (self.head + alignment - 1) & !(alignment - 1);

        let wraps = offset + size > self.size;

        if wraps {
            offset = 0;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.buffer
                    .allocation
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr()
                    .byte_add(offset as usize)
                    .cast(),
                data.len(),
            );
        }

        let allocation = RingStagingBufferAllocation { size, offset };
        self.frame_allocations[frame_index].push(allocation.clone());

        if wraps {
            self.head = size;
        } else {
            self.head = offset + size;
        }

        if self.head >= self.size {
            self.head = 0;
        }

        Ok(allocation)
    }

    pub fn transfer(
        &mut self,
        context: Arc<RenderingContext>,
        cmd: vk::CommandBuffer,
        allocation: RingStagingBufferAllocation,
        dst_buf: &AllocatedBuffer,
        dst_stage_mask: vk::PipelineStageFlags2,
        dst_access_mask: vk::AccessFlags2,
    ) {
        unsafe {
            context.device.cmd_copy_buffer(
                cmd,
                self.buffer.buffer,
                dst_buf.buffer,
                &[vk::BufferCopy::default()
                    .dst_offset(0)
                    .src_offset(allocation.offset)
                    .size(allocation.size)],
            );
            context.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(dst_stage_mask)
                        .dst_access_mask(dst_access_mask)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(dst_buf.buffer)
                        .size(dst_buf.allocation.size()),
                ]),
            );
        }
    }

    pub fn retire_frame(&mut self, frame_index: usize) -> Result<()> {
        self.frame_allocations[frame_index].clear();
        Ok(())
    }
}
