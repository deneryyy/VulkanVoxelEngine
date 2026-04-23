pub mod camera;
pub mod context;
pub mod descriptor;
pub mod model;
pub mod quad;
pub mod staging;
pub mod swapchain;
pub mod terrain;
pub mod texture;
pub mod vertex;

use std::{sync::Arc, u32};

use anyhow::{Ok, Result};
use glam::{Mat4, Vec4};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use parking_lot::Mutex;
use winit::window::Window;

use crate::app::{
    engine::renderer::{
        context::{AllocatedBuffer, RenderingContext},
        descriptor::{DescriptorAllocator, DescriptorLayoutBuilder, PoolSizeRatio},
        staging::RingStagingBuffer,
        swapchain::Swapchain,
        terrain::TerrainRenderer,
        texture::Texture,
    },
    game::world::World,
};
use ash::vk::{self, CommandBufferUsageFlags};

pub struct Frame {
    ray_traversal_descriptors: vk::DescriptorSet,
    ray_traversal_descriptor_layout: vk::DescriptorSetLayout,
    ray_traversal_uniform_buffer: AllocatedBuffer,
    vertex_terrain_descriptors: vk::DescriptorSet,
    vertex_terrain_descriptor_layout: vk::DescriptorSetLayout,
    uniform_buffer: AllocatedBuffer,
    in_flight_fence: vk::Fence,
    image_available_semaphore: vk::Semaphore,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
}

#[repr(C)]
struct RayTraversalUBO {
    inv_view: glam::Mat4,
    inv_proj: glam::Mat4,
    camera_pos: glam::Vec4,
}

#[repr(C)]
struct UniformBufferObject {
    view: glam::Mat4,
    proj: glam::Mat4,
}

pub struct Renderer {
    terrain_renderer: TerrainRenderer,
    texture_descriptors: vk::DescriptorSet,
    texture_descriptor_layout: vk::DescriptorSetLayout,
    ring_staging_buffer: RingStagingBuffer,
    allocator: Arc<Mutex<Allocator>>,
    world: Arc<World>,
    textures: Vec<Texture>,
    global_descriptor_allocator: DescriptorAllocator,
    render_finished_semaphores: Vec<vk::Semaphore>,
    frame_number: usize,
    frames: Vec<Frame>,
    swapchain: Swapchain,
    context: Arc<RenderingContext>,
}

pub const FRAME_OVERLAP: usize = 2;

impl Renderer {
    pub fn new(
        context: Arc<RenderingContext>,
        window: Arc<Window>,
        world: Arc<World>,
    ) -> Result<Self> {
        let mut swapchain = Swapchain::new(context.clone(), window.clone())?;
        swapchain.resize()?;

        let allocator = Arc::new(Mutex::new(Allocator::new(&AllocatorCreateDesc {
            instance: context.instance.clone(),
            device: context.device.clone(),
            physical_device: context.physical_device.handle,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?));

        let mut textures: Vec<Texture> = Vec::new();
        let grass_texture = Texture::load_texture_from_file(
            "textures/grass.png",
            context.clone(),
            &mut allocator.lock(),
        )?;
        let stone_texture = Texture::load_texture_from_file(
            "textures/stone.webp",
            context.clone(),
            &mut allocator.lock(),
        )?;
        textures.push(grass_texture);
        textures.push(stone_texture);
        let supported_textures: u32 = textures.len() as u32;

        let sizes = Vec::from([
            PoolSizeRatio {
                desc_type: vk::DescriptorType::UNIFORM_BUFFER,
                ratio: 2,
            },
            PoolSizeRatio {
                desc_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                ratio: supported_textures,
            },
            PoolSizeRatio {
                desc_type: vk::DescriptorType::STORAGE_IMAGE,
                ratio: 1,
            },
        ]);
        let global_descriptor_allocator =
            DescriptorAllocator::new(&context.device, FRAME_OVERLAP as u32 * 2 + 1, sizes)?;

        let mut texture_descriptor_layout_builed = DescriptorLayoutBuilder::new();
        texture_descriptor_layout_builed.add_binding(
            0,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            supported_textures,
            vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
            vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        );

        let texture_descriptor_layout = texture_descriptor_layout_builed.build(&context.device)?;
        let texture_descriptors = global_descriptor_allocator.allocate(
            &context.device,
            texture_descriptor_layout,
            &[supported_textures],
        )?;
        let image_infos: Vec<vk::DescriptorImageInfo> = textures
            .iter()
            .map(|tex| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(tex.image.image_view)
                    .sampler(tex.sampler)
            })
            .collect();

        let tex_write = vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(texture_descriptors)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);
        unsafe {
            context.device.update_descriptor_sets(&[tex_write], &[]);
            let mut render_finished_semaphores: Vec<vk::Semaphore> =
                Vec::with_capacity(swapchain.images.len() as usize);
            for _ in 0..swapchain.images.len() {
                let render_finished_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                render_finished_semaphores.push(render_finished_semaphore);
            }

            let mut frames: Vec<Frame> = Vec::with_capacity(FRAME_OVERLAP);
            for _ in 0..FRAME_OVERLAP {
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
                    "Uniform buffer",
                    &mut allocator.lock(),
                    std::mem::size_of::<UniformBufferObject>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;
                let ray_traversal_uniform_buffer = context.create_buffer(
                    "Ray traversal uniform buffer",
                    &mut allocator.lock(),
                    std::mem::size_of::<RayTraversalUBO>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;
                let mut vertex_terrain_descriptor_layout_builder = DescriptorLayoutBuilder::new();
                let mut ray_traversal_descriptor_layout_builder = DescriptorLayoutBuilder::new();
                vertex_terrain_descriptor_layout_builder.add_binding(
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    1,
                    vk::ShaderStageFlags::VERTEX,
                    vk::DescriptorBindingFlags::empty(),
                );
                ray_traversal_descriptor_layout_builder.add_binding(
                    0,
                    vk::DescriptorType::STORAGE_IMAGE,
                    1,
                    vk::ShaderStageFlags::COMPUTE,
                    vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                );
                ray_traversal_descriptor_layout_builder.add_binding(
                    1,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    1,
                    vk::ShaderStageFlags::COMPUTE,
                    vk::DescriptorBindingFlags::empty(),
                );
                let vertex_terrain_descriptor_layout =
                    vertex_terrain_descriptor_layout_builder.build(&context.device)?;
                let ray_traversal_descriptor_layout =
                    ray_traversal_descriptor_layout_builder.build(&context.device)?;

                let vertex_terrain_descriptors = global_descriptor_allocator.allocate(
                    &context.device,
                    vertex_terrain_descriptor_layout,
                    &[],
                )?;
                let ray_traversal_descriptors = global_descriptor_allocator.allocate(
                    &context.device,
                    ray_traversal_descriptor_layout,
                    &[],
                )?;

                let mut desc_writes: Vec<vk::WriteDescriptorSet> = Vec::new();
                let uniform_desc_info = &[vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffer.buffer)
                    .range(std::mem::size_of::<UniformBufferObject>() as u64)];
                desc_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_binding(0)
                        .dst_set(vertex_terrain_descriptors)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(uniform_desc_info),
                );

                desc_writes.push(tex_write);

                let draw_image_desc_info = &[vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(swapchain.draw_image.image_view)];
                desc_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_binding(0)
                        .dst_set(ray_traversal_descriptors)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .image_info(draw_image_desc_info),
                );
                let rt_uniform_desc_info = &[vk::DescriptorBufferInfo::default()
                    .buffer(ray_traversal_uniform_buffer.buffer)
                    .range(std::mem::size_of::<RayTraversalUBO>() as u64)];
                desc_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_binding(1)
                        .dst_set(ray_traversal_descriptors)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(rt_uniform_desc_info),
                );

                context.device.update_descriptor_sets(&desc_writes, &[]);

                frames.push(Frame {
                    ray_traversal_descriptors,
                    ray_traversal_descriptor_layout,
                    ray_traversal_uniform_buffer,
                    vertex_terrain_descriptors,
                    vertex_terrain_descriptor_layout,
                    uniform_buffer,
                    image_available_semaphore,
                    in_flight_fence,
                    command_buffer,
                    command_pool,
                });
            }

            let vertex_terrain_descriptor_set_layouts = frames
                .iter()
                .map(|frame| return frame.vertex_terrain_descriptor_layout)
                .collect::<Vec<_>>();
            let ray_traversal_descriptor_set_layouts = frames
                .iter()
                .map(|frame| return frame.ray_traversal_descriptor_layout)
                .collect::<Vec<_>>();

            let render_distance;
            {
                let player = world.player.read();
                render_distance = player.render_distance;
            }
            Ok(Self {
                terrain_renderer: TerrainRenderer::new(
                    world.clone(),
                    render_distance,
                    textures.len() as u32,
                    context.clone(),
                    allocator.clone(),
                    &swapchain,
                    &vertex_terrain_descriptor_set_layouts,
                    &texture_descriptor_layout,
                    &ray_traversal_descriptor_set_layouts,
                )?,
                texture_descriptor_layout,
                texture_descriptors,
                ring_staging_buffer: RingStagingBuffer::new(
                    context.clone(),
                    &mut allocator.lock(),
                    64 * 1024 * 1024,
                )?,
                allocator: allocator.clone(),
                world,
                textures,
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

    fn upload_uniform_buffer(world: Arc<World>, swapchain: &Swapchain, frame: &Frame) {
        unsafe {
            let player = world.player.read();
            let mut proj = glam::Mat4::perspective_rh(
                45.0,
                swapchain.draw_image.extent.width as f32
                    / swapchain.draw_image.extent.height as f32,
                0.1,
                1000.0,
            );
            proj.y_axis.y *= -1.0;

            let view = player.camera.get_view_matrix();
            let ubo = UniformBufferObject { view, proj };
            let ubo_ptr = frame
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr();
            std::ptr::copy_nonoverlapping(&ubo, ubo_ptr.cast(), 1);

            let rtubo = RayTraversalUBO {
                inv_view: view.inverse(),
                inv_proj: proj.inverse(),
                camera_pos: player.camera.position.extend(1.0),
            };
            let rtubo_ptr = frame
                .ray_traversal_uniform_buffer
                .allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr();
            std::ptr::copy_nonoverlapping(&rtubo, rtubo_ptr.cast(), 1);
        }
    }

    pub fn render(&mut self) -> Result<()> {
        unsafe {
            let frame_index = self.frame_number % FRAME_OVERLAP;
            let frame = &self.frames[frame_index];

            // Wait until gpu finished rendering last frame
            self.context
                .device
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
            self.ring_staging_buffer.retire_frame(frame_index)?;

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

            Self::upload_uniform_buffer(self.world.clone(), &self.swapchain, frame);

            self.terrain_renderer.receive_chunks(
                &mut self.ring_staging_buffer,
                frame,
                frame_index,
            )?;

            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.draw_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );

            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.depth_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
            );

            let extent = vk::Extent2D::default()
                .width(self.swapchain.draw_image.extent.width)
                .height(self.swapchain.draw_image.extent.height);

            self.context.begin_rendering(
                frame.command_buffer,
                self.swapchain.draw_image.image_view,
                self.swapchain.depth_image.image_view,
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
                    .height(extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );

            self.context.device.cmd_set_scissor(
                frame.command_buffer,
                0,
                &[vk::Rect2D::default().extent(extent)],
            );

            self.terrain_renderer
                .record_commands(frame, &self.frames, extent, &self.swapchain, self.texture_descriptors);

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
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.render_finished_semaphores
                .drain(..)
                .for_each(|semaphore| {
                    self.context.device.destroy_semaphore(semaphore, None);
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
                    .destroy_buffer(&mut self.allocator.lock(), &mut frame.uniform_buffer);
                self.context.destroy_buffer(
                    &mut self.allocator.lock(),
                    &mut frame.ray_traversal_uniform_buffer,
                );
                self.context
                    .device
                    .destroy_descriptor_set_layout(frame.vertex_terrain_descriptor_layout, None);

                self.context
                    .device
                    .destroy_descriptor_set_layout(frame.ray_traversal_descriptor_layout, None);
            });
            self.context
                .device
                .destroy_descriptor_set_layout(self.texture_descriptor_layout, None);

            self.global_descriptor_allocator
                .destroy_pool(&self.context.device);

            self.context.destroy_buffer(
                &mut self.allocator.lock(),
                &mut self.ring_staging_buffer.buffer,
            );
            for texture in &mut self.textures {
                Texture::destroy_texture(self.context.clone(), &mut self.allocator.lock(), texture);
            }
        }
    }
}
