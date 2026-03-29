pub mod context;
pub mod descriptor;
pub mod swapchain;
pub mod model;
pub mod vertex;

use std::{sync::Arc, u32};

use anyhow::{Ok, Result};
use glam::Mat4;
use gpu_allocator::{
    AllocatorDebugSettings,
    vulkan::{Allocator, AllocatorCreateDesc},
};
use std::result::Result::Ok as stdOk;
use winit::window::Window;

use crate::app::engine::{
    renderer::{
        context::{AllocatedBuffer, AllocatedImage, RenderingContext},
        descriptor::{DescriptorAllocator, DescriptorLayoutBuilder, PoolSizeRatio},
        swapchain::Swapchain,
        vertex::Vertex,
    },
    resources,
};
use ash::vk::{self, CommandBufferUsageFlags, ExtendsDescriptorSetLayoutCreateInfo};

pub struct Frame {
    uniform_buffer_descriptors: vk::DescriptorSet,
    uniform_buffer_descriptor_layout: vk::DescriptorSetLayout,
    uniform_buffer: AllocatedBuffer,
    in_flight_fence: vk::Fence,
    image_available_semaphore: vk::Semaphore,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
}

struct GPUMeshBuffers {
    vertex_buffer_address: vk::DeviceAddress,
    index_buffer: AllocatedBuffer,
    vertex_buffer: AllocatedBuffer,
}

#[repr(C, align(16))]
struct GPUDrawPushConstants {
    vertex_buffer: vk::DeviceAddress,
}

struct UniformBufferObject {
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
}

pub struct Renderer {
    allocator: Allocator,
    gpu_mesh_buffers: GPUMeshBuffers,
    imm_fence: vk::Fence,
    imm_command_buffer: vk::CommandBuffer,
    imm_command_pool: vk::CommandPool,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    global_descriptor_allocator: DescriptorAllocator,
    render_finished_semaphores: Vec<vk::Semaphore>,
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

        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: context.instance.clone(),
            device: context.device.clone(),
            physical_device: context.physical_device.handle,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let sizes = Vec::from([PoolSizeRatio {
            desc_type: vk::DescriptorType::UNIFORM_BUFFER,
            ratio: 1,
        }]);
        let global_descriptor_allocator = DescriptorAllocator::new(&context.device, FRAME_OVERLAP as u32, sizes)?;

        unsafe {
            let mut render_finished_semaphores: Vec<vk::Semaphore> = Vec::with_capacity(swapchain.images.len() as usize);
            for i in 0..swapchain.images.len() {
                let render_finished_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                render_finished_semaphores.push(render_finished_semaphore);
            }

            let mut frames: Vec<Frame> = Vec::with_capacity(FRAME_OVERLAP);
            for i in 0..FRAME_OVERLAP {
                let image_available_semaphore = context
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

                let uniform_buffer = context.create_buffer(
                    &mut allocator,
                    std::mem::size_of::<UniformBufferObject>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;
                let mut descriptor_layout_builder = DescriptorLayoutBuilder::new();
                descriptor_layout_builder.add_binding(0, vk::DescriptorType::UNIFORM_BUFFER);

                let uniform_buffer_descriptor_layout = descriptor_layout_builder
                    .build(&context.device, vk::ShaderStageFlags::VERTEX)?;

                let uniform_buffer_descriptors = global_descriptor_allocator
                    .allocate(&context.device, uniform_buffer_descriptor_layout)?;

                context.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_binding(0)
                        .dst_set(uniform_buffer_descriptors)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(uniform_buffer.buffer)
                            .range(std::mem::size_of::<UniformBufferObject>() as u64)])],
                    &[],
                );

                frames.push(Frame {
                    uniform_buffer_descriptors,
                    uniform_buffer_descriptor_layout,
                    uniform_buffer,
                    image_available_semaphore,
                    in_flight_fence,
                    command_buffer,
                    command_pool,
                });
            }

            let descriptor_set_layouts = frames
                .iter()
                .map(|frame| return frame.uniform_buffer_descriptor_layout)
                .collect::<Vec<_>>();
            let pipeline_layout = context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&descriptor_set_layouts)
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .offset(0)
                        .size(std::mem::size_of::<GPUDrawPushConstants>() as u32)
                        .stage_flags(vk::ShaderStageFlags::VERTEX)]),
                None,
            )?;

            let vertex_shader = Self::load_shader_module("shader.vert.spv", &context.device)?;
            let fragment_shader = Self::load_shader_module("shader.frag.spv", &context.device)?;
            let pipeline = context.create_graphics_pipeline(
                vk::Extent2D::default()
                    .width(swapchain.draw_image.extent.width)
                    .height(swapchain.draw_image.extent.height),
                swapchain.draw_image.image_format,
                vertex_shader,
                fragment_shader,
                pipeline_layout,
                Default::default(),
            )?;
            context.device.destroy_shader_module(vertex_shader, None);
            context.device.destroy_shader_module(fragment_shader, None);

            let imm_command_pool = context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.queue_families.graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;

            let &imm_command_buffer = context
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(imm_command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?
                .iter()
                .next()
                .unwrap();

            let imm_fence = context.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;

            let vertices = &[
                Vertex::new([-0.5, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]),
                Vertex::new([0.5, -0.5, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
                Vertex::new([0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]),
                Vertex::new([-0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ];
            let indices: &[u32] = &[0, 1, 2, 2, 3, 0];

            let vertex_buffer_size = (vertices.len() * std::mem::size_of::<Vertex>()) as u64;

            let vertex_buffer = context.create_buffer(
                &mut allocator,
                vertex_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;

            let index_buffer_size = (indices.len() * std::mem::size_of::<u32>()) as u64;
            let index_buffer = context.create_buffer(
                &mut allocator,
                index_buffer_size,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;

            let mut staging = context.create_buffer(
                &mut allocator,
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

            let cmd = Self::begin_immediate_submit(context.clone(), imm_fence, imm_command_buffer)?;

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

            Self::end_immediate_submit(context.clone(), imm_fence, imm_command_buffer)?;
            let vertex_buffer_address = context.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer),
            );
            context.destroy_buffer(&mut allocator, &mut staging);
            let gpu_mesh_buffers = GPUMeshBuffers {
                vertex_buffer_address,
                vertex_buffer,
                index_buffer,
            };

            Ok(Self {
                allocator,
                gpu_mesh_buffers,
                imm_fence,
                imm_command_buffer,
                imm_command_pool,
                pipeline,
                pipeline_layout,
                global_descriptor_allocator,
                render_finished_semaphores,
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
        let binary_data = resources::load_spirv(path)?;

        unsafe {
            Ok(device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&binary_data),
                None,
            )?)
        }
    }

    fn begin_immediate_submit(
        context: Arc<RenderingContext>,
        imm_fence: vk::Fence,
        imm_command_buffer: vk::CommandBuffer,
    ) -> Result<vk::CommandBuffer> {
        unsafe {
            context.device.reset_fences(&[imm_fence])?;
            context
                .device
                .reset_command_buffer(imm_command_buffer, vk::CommandBufferResetFlags::empty())?;
            context.device.begin_command_buffer(
                imm_command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            Ok(imm_command_buffer)
        }
    }

    fn end_immediate_submit(
        context: Arc<RenderingContext>,
        imm_fence: vk::Fence,
        imm_command_buffer: vk::CommandBuffer,
    ) -> Result<()> {
        unsafe {
            context.device.end_command_buffer(imm_command_buffer)?;

            context.device.queue_submit2(
                context.queues[context.queue_families.graphics as usize],
                &[vk::SubmitInfo2::default().command_buffer_infos(&[
                    vk::CommandBufferSubmitInfo::default().command_buffer(imm_command_buffer),
                ])],
                imm_fence,
            )?;

            context
                .device
                .wait_for_fences(&[imm_fence], true, u64::MAX)?;
            Ok(())
        }
    }

    pub fn render(&mut self) -> Result<()> {
        unsafe {
            let frame = &mut self.frames[self.frame_number % FRAME_OVERLAP];

            let ubo = UniformBufferObject {
                model: glam::Mat4::from_rotation_x(45.0 + (self.frame_number as f32) * 0.01),
                view: glam::Mat4::look_at_lh(
                    glam::Vec3 {
                        x: 2.0,
                        y: 2.0,
                        z: 2.0,
                    },
                    glam::Vec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    glam::Vec3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                ),
                proj: glam::Mat4::perspective_lh(
                    45.0,
                    self.swapchain.draw_image.extent.width as f32
                        / self.swapchain.draw_image.extent.height as f32,
                    0.0,
                    1000.0,
                ),
            };
            let ubo_ptr = frame
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr();
            std::ptr::copy_nonoverlapping(&ubo, ubo_ptr.cast(), 1);

            // Wait until gpu finished rendering last frame
            self.context
                .device
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;

            if self.swapchain.is_dirty {
                self.swapchain.resize()?;
                self.swapchain.is_dirty = false;
            }

            let image_index_result = self
                .swapchain
                .acquire_next_image(frame.image_available_semaphore);

            if image_index_result.is_err() {
                self.swapchain.resize()?;
                return Ok(());
            }
            let image_index = image_index_result?;

            self.context.device.reset_fences(&[frame.in_flight_fence])?;

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
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );

            let extent = vk::Extent2D::default()
                .width(self.swapchain.draw_image.extent.width)
                .height(self.swapchain.draw_image.extent.height);

            self.context.begin_rendering(
                frame.command_buffer,
                self.swapchain.draw_image.image_view,
                vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
                vk::Rect2D::default().extent(extent),
            );

            self.context.device.cmd_set_viewport(
                frame.command_buffer,
                0,
                &[vk::Viewport::default()
                    .width(extent.width as f32)
                    .height(extent.height as f32)],
            );

            self.context.device.cmd_set_scissor(
                frame.command_buffer,
                0,
                &[vk::Rect2D::default().extent(extent)],
            );

            self.context.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            let push_constants = GPUDrawPushConstants {
                vertex_buffer: self.gpu_mesh_buffers.vertex_buffer_address,
            };
            self.context.device.cmd_push_constants(
                frame.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const _ as *const u8,
                    std::mem::size_of::<GPUDrawPushConstants>(),
                ),
            );
            self.context.device.cmd_bind_index_buffer(
                frame.command_buffer,
                self.gpu_mesh_buffers.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );

            self.context.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[frame.uniform_buffer_descriptors],
                &[],
            );

            self.context
                .device
                .cmd_draw_indexed(frame.command_buffer, 6, 1, 0, 0, 0);

            self.context.device.cmd_end_rendering(frame.command_buffer);

            //transition the draw image and the swapchain image into their correct transfer layouts
            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.draw_image.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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

            let render_finished_semaphore = self.render_finished_semaphores[image_index as usize];

            self.context.device.queue_submit2(
                self.context.queues[self.context.queue_families.graphics as usize],
                &[vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(frame.image_available_semaphore)
                        .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(render_finished_semaphore)
                        .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)])
                    .command_buffer_infos(&[
                        vk::CommandBufferSubmitInfo::default().command_buffer(frame.command_buffer)
                    ])],
                frame.in_flight_fence,
            )?;

            self.swapchain
                .present(image_index, render_finished_semaphore)?;

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

            self.render_finished_semaphores.drain(..).for_each(|semaphore| {
                self.context
                    .device
                    .destroy_semaphore(semaphore, None);
            });

            self.frames.drain(..).for_each(|mut frame| {
                self.context
                    .device
                    .destroy_semaphore(frame.image_available_semaphore, None);
               self.context
                    .device
                    .destroy_fence(frame.in_flight_fence, None);
                self.context
                    .device
                    .free_command_buffers(frame.command_pool, &[frame.command_buffer]);
                self.context
                    .device
                    .destroy_command_pool(frame.command_pool, None);
                self.context
                    .destroy_buffer(&mut self.allocator, &mut frame.uniform_buffer);
                self.context
                    .device
                    .destroy_descriptor_set_layout(frame.uniform_buffer_descriptor_layout, None);
            });

            self.global_descriptor_allocator
                .destroy_pool(&self.context.device);

            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.context.device.destroy_pipeline(self.pipeline, None);

            self.context.device.destroy_fence(self.imm_fence, None);
            self.context
                .device
                .destroy_command_pool(self.imm_command_pool, None);

            self.context.destroy_buffer(
                &mut self.allocator,
                &mut self.gpu_mesh_buffers.vertex_buffer,
            );
            self.context
                .destroy_buffer(&mut self.allocator, &mut self.gpu_mesh_buffers.index_buffer);
            /*
            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            */
        }
    }
}
