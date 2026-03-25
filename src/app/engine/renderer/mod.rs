pub mod context;
pub mod deletion_queue;
pub mod swapchain;

use std::sync::Arc;

use anyhow::{Ok, Result};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::window::Window;

use crate::app::engine::renderer::{
    context::{AllocatedImage, RenderingContext},
    deletion_queue::DeletionQueue,
    swapchain::Swapchain,
};
use ash::vk::{self, CommandBufferUsageFlags};

pub struct Frame {
    deletion_queue: DeletionQueue,
    in_flight_fence: vk::Fence,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
}

pub struct Renderer {
    deletion_queue: DeletionQueue,
    frame_number: usize,
    frames: Vec<Frame>,
    swapchain: Swapchain,
    context: Arc<RenderingContext>,
}

const FRAME_OVERLAP: usize = 2;

impl Renderer {
    pub fn new(context: Arc<RenderingContext>, window: Arc<Window>) -> Result<Self> {
        let mut swapchain = Swapchain::new(context.clone(), window.clone())?;
        swapchain.resize()?;
        unsafe {
            let mut frames: Vec<Frame> = Vec::with_capacity(FRAME_OVERLAP);
            for i in 0..FRAME_OVERLAP {
                let image_available_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                let render_finished_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                let in_flight_fence = context.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?;
                let command_pool = context.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(context.queue_families.graphics)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )?;

                let &command_buffer = context
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )?
                    .iter()
                    .next()
                    .unwrap();

                frames.push(Frame {
                    deletion_queue: DeletionQueue::new(),
                    image_available_semaphore,
                    render_finished_semaphore,
                    in_flight_fence,
                    command_buffer,
                    command_pool,
                });
            }

            Ok(Self {
                deletion_queue: DeletionQueue::new(),
                frame_number: 0,
                frames,
                swapchain,
                context,
            })
        }
    }

    pub fn resize(&mut self) -> Result<()> {
        self.swapchain.resize()
    }

    pub fn render(&mut self) -> Result<()> {
        unsafe {
            let frame = &mut self.frames[self.frame_number % FRAME_OVERLAP];

            // Wait until gpu finished rendering last frame
            self.context
                .device
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
            frame.deletion_queue.flush();

            self.context.device.reset_fences(&[frame.in_flight_fence])?;

            if self.swapchain.is_dirty {
                self.swapchain.resize()?;
            }

            let image_index = self
                .swapchain
                .acquire_next_image(frame.image_available_semaphore)?;

            self.context
                .device
                .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())?;

            self.context.device.begin_command_buffer(
                frame.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // transition our main draw image into general layout so we can write into it
            // we will overwrite it all so we dont care about what was the older layout
            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.draw_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::ImageAspectFlags::COLOR,
            );

            self.context.device.cmd_clear_color_image(
                frame.command_buffer,
                self.swapchain.draw_image.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                },
                &[vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .layer_count(vk::REMAINING_ARRAY_LAYERS)
                    .base_array_layer(0)
                    .level_count(vk::REMAINING_MIP_LEVELS)],
            );

            //transition the draw image and the swapchain image into their correct transfer layouts
            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.draw_image.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );
            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.images[image_index as usize],
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );

            // execute a copy from the draw image into the swapchain
            self.context.copy_image_to_image(
                frame.command_buffer,
                self.swapchain.draw_image.image,
                self.swapchain.images[image_index as usize],
                vk::Extent2D::default()
                    .width(self.swapchain.draw_image.extent.width)
                    .height(self.swapchain.draw_image.extent.height),
                self.swapchain.extent,
            );

            // set swapchain image layout to Present so we can show it on the screen
            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.images[image_index as usize],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::ImageAspectFlags::COLOR,
            );
            self.context
                .device
                .end_command_buffer(frame.command_buffer)?;

            self.context.device.queue_submit2(
                self.context.queues[self.context.queue_families.graphics as usize],
                &[vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(frame.image_available_semaphore)
                        .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(frame.render_finished_semaphore)
                        .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)])
                    .command_buffer_infos(&[
                        vk::CommandBufferSubmitInfo::default().command_buffer(frame.command_buffer)
                    ])],
                frame.in_flight_fence,
            )?;

            self.swapchain
                .present(image_index, frame.render_finished_semaphore)?;

            self.frame_number += 1;

            Ok(())
        }
    }

    //fn get_current_frame(&self) -> Result<Frame> {
    //}
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.frames.drain(..).for_each(|frame| {
                self.context
                    .device
                    .destroy_semaphore(frame.image_available_semaphore, None);
                self.context
                    .device
                    .destroy_semaphore(frame.render_finished_semaphore, None);
                self.context
                    .device
                    .destroy_fence(frame.in_flight_fence, None);
                self.context
                    .device
                    .free_command_buffers(frame.command_pool, &[frame.command_buffer]);
                self.context
                    .device
                    .destroy_command_pool(frame.command_pool, None);
                self.deletion_queue.flush();
            });

            self.deletion_queue.flush();
            /*
            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            */
        }
    }
}
