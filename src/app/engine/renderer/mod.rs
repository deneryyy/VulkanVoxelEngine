pub mod context;
pub mod swapchain;

use std::sync::Arc;

use anyhow::{Ok, Result};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::window::Window;

use crate::app::engine::{
    renderer::{
        context::{AllocatedImage, RenderingContext},
        swapchain::Swapchain,
    },
    resources,
};
use ash::vk::{self, CommandBufferUsageFlags, ExtendsDescriptorSetLayoutCreateInfo};

pub struct DescriptorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl<'a> DescriptorLayoutBuilder<'a> {
    fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    fn add_binding(&mut self, binding: u32, desc_type: vk::DescriptorType) {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_count(1)
                .descriptor_type(desc_type),
        );
    }

    fn clear(&mut self) {
        self.bindings.clear();
    }

    fn build(
        &mut self,
        device: &ash::Device,
        shader_stages: vk::ShaderStageFlags,
    ) -> Result<vk::DescriptorSetLayout> {
        for b in &mut self.bindings {
            b.stage_flags |= shader_stages
        }

        unsafe {
            Ok(device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&self.bindings)
                    .flags(vk::DescriptorSetLayoutCreateFlags::empty()),
                None,
            )?)
        }
    }
}

struct DescriptorAllocator {
    pool: vk::DescriptorPool,
}
struct PoolSizeRatio {
    desc_type: vk::DescriptorType,
    ratio: u32,
}

impl DescriptorAllocator {
    fn new(device: &ash::Device, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) -> Result<Self> {
        let mut pool_sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for ratio in pool_ratios {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: ratio.desc_type,
                descriptor_count: ratio.ratio * max_sets,
            });
        }

        unsafe {
            let pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::empty())
                    .max_sets(max_sets)
                    .pool_sizes(&pool_sizes),
                None,
            )?;

            Ok(Self { pool })
        }
    }

    fn clear_descriptors(&self, device: &ash::Device) {
        unsafe {
            device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty());
        }
    }

    fn destroy_pool(&self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
        }
    }

    fn allocate(
        &self,
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        unsafe {
            Ok(*device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.pool)
                        .set_layouts(&[layout]),
                )?
                .iter()
                .next()
                .unwrap())
        }
    }
}

pub struct Frame {
    in_flight_fence: vk::Fence,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
}

pub struct Renderer {
    gradient_pipeline: vk::Pipeline,
    gradient_pipeline_layout: vk::PipelineLayout,
    global_descriptor_allocator: DescriptorAllocator,
    draw_image_descriptors: vk::DescriptorSet,
    draw_image_descriptor_layout: vk::DescriptorSetLayout,
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
                    image_available_semaphore,
                    render_finished_semaphore,
                    in_flight_fence,
                    command_buffer,
                    command_pool,
                });
            }

            let sizes = Vec::from([PoolSizeRatio {
                desc_type: vk::DescriptorType::STORAGE_IMAGE,
                ratio: 1,
            }]);

            let global_descriptor_allocator = DescriptorAllocator::new(&context.device, 10, sizes)?;
            let mut descriptor_layout_builder = DescriptorLayoutBuilder::new();
            descriptor_layout_builder.add_binding(0, vk::DescriptorType::STORAGE_IMAGE);

            let draw_image_descriptor_layout =
                descriptor_layout_builder.build(&context.device, vk::ShaderStageFlags::COMPUTE)?;

            let draw_image_descriptors = global_descriptor_allocator
                .allocate(&context.device, draw_image_descriptor_layout)?;

            context.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .dst_set(draw_image_descriptors)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(swapchain.draw_image.image_view)])],
                &[],
            );

            let gradient_pipeline_layout = context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[draw_image_descriptor_layout]),
                None,
            )?;

            let module = Self::load_shader_module("gradient.comp.spv", &context.device)?;

            let gradient_pipeline = *context
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .layout(gradient_pipeline_layout)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(module)
                                .name(c"main"),
                        )],
                    None,
                )
                .unwrap()
                .iter()
                .next()
                .unwrap();

            context.device.destroy_shader_module(module, None);

            Ok(Self {
                gradient_pipeline,
                gradient_pipeline_layout,
                global_descriptor_allocator,
                draw_image_descriptor_layout,
                draw_image_descriptors,
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

    fn load_shader_module(path: &str, device: &ash::Device) -> Result<vk::ShaderModule> {
        let binary_data = resources::load_binary_u32(path)?;

        unsafe {
            Ok(device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&binary_data),
                None,
            )?)
        }
    }

    pub fn render(&mut self) -> Result<()> {
        unsafe {
            let frame = &mut self.frames[self.frame_number % FRAME_OVERLAP];

            // Wait until gpu finished rendering last frame
            self.context
                .device
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;

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

            /*
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
            */
            self.context.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_pipeline,
            );
            self.context.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.gradient_pipeline_layout,
                0,
                &[self.draw_image_descriptors],
                &[],
            );
            self.context.device.cmd_dispatch(
                frame.command_buffer,
                self.swapchain.draw_image.extent.width / 16,
                self.swapchain.draw_image.extent.height / 16,
                1,
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
            });

            self.global_descriptor_allocator
                .destroy_pool(&self.context.device);
            self.context
                .device
                .destroy_descriptor_set_layout(self.draw_image_descriptor_layout, None);

            self.context
                .device
                .destroy_pipeline_layout(self.gradient_pipeline_layout, None);
            self.context
                .device
                .destroy_pipeline(self.gradient_pipeline, None);
            /*
            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            */
        }
    }
}
