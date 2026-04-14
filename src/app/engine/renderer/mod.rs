pub mod camera;
pub mod context;
pub mod descriptor;
pub mod model;
pub mod quad;
pub mod swapchain;
pub mod texture;
pub mod vertex;

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    u32,
};

use anyhow::{Ok, Result, anyhow};
use glam::{IVec3, Mat4, Vec3};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::result::Result::Ok as stdOk;
use winit::window::Window;

use crate::app::{
    engine::{
        renderer::{
            context::{AllocatedBuffer, RenderingContext},
            descriptor::{DescriptorAllocator, DescriptorLayoutBuilder, PoolSizeRatio},
            swapchain::Swapchain,
            texture::Texture,
            vertex::Vertex,
        },
        resources,
    },
    game::world::{ChunkMessage, World},
};
use ash::vk::{self, CommandBufferUsageFlags};

pub struct Frame {
    descriptors: vk::DescriptorSet,
    descriptor_layout: vk::DescriptorSetLayout,
    uniform_buffer: AllocatedBuffer,
    in_flight_fence: vk::Fence,
    image_available_semaphore: vk::Semaphore,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
}

#[repr(C, align(16))]
struct GPUDrawPushConstants {
    vertex_buffer: vk::DeviceAddress,
    matrix_buffer: vk::DeviceAddress,
    material_id: u32,
    render_object_id: u32,
}

struct UniformBufferObject {
    view: glam::Mat4,
    proj: glam::Mat4,
}

pub struct AllocatedChunkMesh {
    index_count: u32,
    first_index: u32,
    material: u32,
    index_buffer: AllocatedBuffer,
    vertex_buffer: AllocatedBuffer,
    vertex_buffer_address: vk::DeviceAddress,
}

struct Matrix {
    model: glam::Mat4,
}

pub struct Renderer {
    ring_staging_buffer: RingStagingBuffer,
    matrix_buffer_address: vk::DeviceAddress,
    matrix_buffer: AllocatedBuffer,
    loaded_chunk_meshes: HashMap<IVec3, Vec<AllocatedChunkMesh>>,
    chunk_mesh_pool: VecDeque<AllocatedChunkMesh>,
    allocator: Allocator,
    world: Arc<World>,
    textures: Vec<Texture>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    global_descriptor_allocator: DescriptorAllocator,
    render_finished_semaphores: Vec<vk::Semaphore>,
    frame_number: usize,
    frames: Vec<Frame>,
    swapchain: Swapchain,
    context: Arc<RenderingContext>,
}

pub struct RingStagingBuffer {
    buffer: AllocatedBuffer,
    size: u64,
    head: u64,
    frame_allocations: Vec<Vec<RingStagingBufferAllocation>>,
}

#[derive(Clone, Copy)]
pub struct RingStagingBufferAllocation {
    size: u64,
    offset: u64,
}

const FRAME_OVERLAP: usize = 2;

impl RingStagingBuffer {
    pub fn new(
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
        size: u64,
    ) -> Result<Self> {
        let buffer = context.create_buffer(
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

    pub fn push<T>(
        &mut self,
        data: &[T],
        frame_index: usize,
    ) -> Result<RingStagingBufferAllocation> {
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

impl Renderer {
    pub fn new(
        context: Arc<RenderingContext>,
        window: Arc<Window>,
        world: Arc<World>,
    ) -> Result<Self> {
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

        let mut textures: Vec<Texture> = Vec::new();
        let grass_texture =
            Texture::load_texture_from_file("textures/grass.png", context.clone(), &mut allocator)?;
        let stone_texture = Texture::load_texture_from_file(
            "textures/stone.webp",
            context.clone(),
            &mut allocator,
        )?;
        textures.push(grass_texture);
        textures.push(stone_texture);
        let supported_textures: u32 = textures.len() as u32;

        let sizes = Vec::from([
            PoolSizeRatio {
                desc_type: vk::DescriptorType::UNIFORM_BUFFER,
                ratio: 1,
            },
            PoolSizeRatio {
                desc_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                ratio: supported_textures,
            },
        ]);
        let global_descriptor_allocator =
            DescriptorAllocator::new(&context.device, FRAME_OVERLAP as u32, sizes)?;

        unsafe {
            let mut render_finished_semaphores: Vec<vk::Semaphore> =
                Vec::with_capacity(swapchain.images.len() as usize);
            for _ in 0..swapchain.images.len() {
                let render_finished_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                render_finished_semaphores.push(render_finished_semaphore);
            }

            let render_distance;
            {
                let player = world.player.read();
                render_distance = player.render_distance;
            }
            let mut chunk_mesh_pool = VecDeque::new();
            for _ in 0..render_distance * render_distance * 4 * textures.len() as u32 {
                // Maximum number of vertices in a chunk
                let vertex_buffer_size = (16 * 16 * 16 * std::mem::size_of::<Vertex>()) as u64;

                let vertex_buffer = context.create_buffer(
                    &mut allocator,
                    vertex_buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    gpu_allocator::MemoryLocation::GpuOnly,
                )?;

                let index_buffer_size = (6 * 16 * 16 * 16 * 3 * std::mem::size_of::<u32>()) as u64;
                let index_buffer = context.create_buffer(
                    &mut allocator,
                    index_buffer_size,
                    vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuOnly,
                )?;

                let vertex_buffer_address = context.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer),
                );
                chunk_mesh_pool.push_back(AllocatedChunkMesh {
                    index_count: 0,
                    first_index: 0,
                    material: 0,
                    index_buffer,
                    vertex_buffer,
                    vertex_buffer_address,
                });
            }

            let matrix_buffer_size = (render_distance as usize
                * render_distance as usize
                * 6
                * std::mem::size_of::<Matrix>()) as u64;
            let matrix_buffer = context.create_buffer(
                &mut allocator,
                matrix_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;
            let matrix_buffer_address = context.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(matrix_buffer.buffer),
            );

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
                    &mut allocator,
                    std::mem::size_of::<UniformBufferObject>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;
                let mut descriptor_layout_builder = DescriptorLayoutBuilder::new();
                descriptor_layout_builder.add_binding(
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    1,
                    vk::ShaderStageFlags::VERTEX,
                    vk::DescriptorBindingFlags::empty(),
                );
                descriptor_layout_builder.add_binding(
                    1,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    supported_textures,
                    vk::ShaderStageFlags::FRAGMENT,
                    vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                        | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                        | vk::DescriptorBindingFlags::PARTIALLY_BOUND,
                );

                let descriptor_layout = descriptor_layout_builder.build(&context.device)?;

                let descriptors = global_descriptor_allocator.allocate(
                    &context.device,
                    descriptor_layout,
                    &[supported_textures],
                )?;

                let mut desc_writes: Vec<vk::WriteDescriptorSet> = Vec::new();
                let uniform_desc_info = &[vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffer.buffer)
                    .range(std::mem::size_of::<UniformBufferObject>() as u64)];
                desc_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_binding(0)
                        .dst_set(descriptors)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(uniform_desc_info),
                );
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
                    .dst_binding(1)
                    .dst_set(descriptors)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&image_infos);
                desc_writes.push(tex_write);

                context.device.update_descriptor_sets(&desc_writes, &[]);

                frames.push(Frame {
                    descriptors,
                    descriptor_layout,
                    uniform_buffer,
                    image_available_semaphore,
                    in_flight_fence,
                    command_buffer,
                    command_pool,
                });
            }

            let descriptor_set_layouts = frames
                .iter()
                .map(|frame| return frame.descriptor_layout)
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

            let vertex_shader =
                resources::load_shader_module("compiled_shaders/shader.vert.spv", &context.device)?;
            let fragment_shader =
                resources::load_shader_module("compiled_shaders/shader.frag.spv", &context.device)?;
            let pipeline = context.create_graphics_pipeline(
                vk::Extent2D::default()
                    .width(swapchain.draw_image.extent.width)
                    .height(swapchain.draw_image.extent.height),
                swapchain.draw_image.image_format,
                swapchain.depth_image.image_format,
                vertex_shader,
                fragment_shader,
                pipeline_layout,
                Default::default(),
            )?;
            context.device.destroy_shader_module(vertex_shader, None);
            context.device.destroy_shader_module(fragment_shader, None);

            Ok(Self {
                matrix_buffer,
                matrix_buffer_address,
                chunk_mesh_pool,
                loaded_chunk_meshes: HashMap::new(),
                ring_staging_buffer: RingStagingBuffer::new(
                    context.clone(),
                    &mut allocator,
                    64 * 1024 * 1024,
                )?,
                allocator,
                world,
                textures,
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

            let ubo = UniformBufferObject {
                view: player.camera.get_view_matrix(),
                proj,
            };
            let ubo_ptr = frame
                .uniform_buffer
                .allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr();
            std::ptr::copy_nonoverlapping(&ubo, ubo_ptr.cast(), 1);
        }
    }

    fn upload_matrix_buffer(
        loaded_chunk_meshes: &HashMap<IVec3, Vec<AllocatedChunkMesh>>,
        ring_staging_buffer: &mut RingStagingBuffer,
        context: Arc<RenderingContext>,
        matrix_buffer: &AllocatedBuffer,
        frame: &Frame,
        frame_index: usize,
    ) -> Result<()> {
        let matrices = loaded_chunk_meshes
            .iter()
            .map(|allocated_chunk_meshes| Matrix {
                model: glam::Mat4::from_translation(Vec3 {
                    x: (allocated_chunk_meshes.0.x * 16) as f32,
                    y: (allocated_chunk_meshes.0.y * 16) as f32,
                    z: (allocated_chunk_meshes.0.z * 16) as f32,
                }),
            })
            .collect::<Vec<_>>();
        let matrix_alloc = ring_staging_buffer.push(&matrices, frame_index)?;
        ring_staging_buffer.transfer(
            context.clone(),
            frame.command_buffer,
            matrix_alloc,
            matrix_buffer,
            vk::PipelineStageFlags2::VERTEX_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
        Ok(())
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

            while let stdOk(chunk_message) = self.world.get_receiver().try_recv() {
                match chunk_message {
                    ChunkMessage::Load(chunk_meshes) => {
                        let mut loaded_chunks = HashMap::new();
                        for chunk_mesh in chunk_meshes.1 {
                            if let Some(mut allocated_chunk_mesh) = self.chunk_mesh_pool.pop_front()
                            {
                                let verts = chunk_mesh.vertices;
                                let indices = Vertex::generate_indices(verts.len());
                                let vert_allocation =
                                    self.ring_staging_buffer.push(&verts, frame_index)?;
                                let index_allocation =
                                    self.ring_staging_buffer.push(&indices, frame_index)?;

                                self.ring_staging_buffer.transfer(
                                    self.context.clone(),
                                    frame.command_buffer,
                                    vert_allocation,
                                    &allocated_chunk_mesh.vertex_buffer,
                                    vk::PipelineStageFlags2::VERTEX_SHADER,
                                    vk::AccessFlags2::SHADER_READ,
                                );
                                self.ring_staging_buffer.transfer(
                                    self.context.clone(),
                                    frame.command_buffer,
                                    index_allocation,
                                    &allocated_chunk_mesh.index_buffer,
                                    vk::PipelineStageFlags2::VERTEX_SHADER,
                                    vk::AccessFlags2::SHADER_READ,
                                );
                                allocated_chunk_mesh.material = chunk_mesh.material;
                                allocated_chunk_mesh.index_count = indices.len() as u32;
                                loaded_chunks
                                    .entry(chunk_mesh.chunk_pos)
                                    .or_insert_with(Vec::new)
                                    .push(allocated_chunk_mesh);
                            }
                        }
                        for loaded_chunk in loaded_chunks {
                            if let Some(old_meshes) = self
                                .loaded_chunk_meshes
                                .insert(loaded_chunk.0, loaded_chunk.1)
                            {
                                for mesh in old_meshes {
                                    self.chunk_mesh_pool.push_back(mesh);
                                }
                            }
                        }
                        Self::upload_matrix_buffer(
                            &self.loaded_chunk_meshes,
                            &mut self.ring_staging_buffer,
                            self.context.clone(),
                            &self.matrix_buffer,
                            frame,
                            frame_index,
                        )?;
                    }
                    ChunkMessage::Unload(pos) => match self.loaded_chunk_meshes.remove(&pos) {
                        Some(allocated_chunk_meshes) => {
                            for allocated_chunk_mesh in allocated_chunk_meshes {
                                self.chunk_mesh_pool.push_back(allocated_chunk_mesh);
                            }
                            Self::upload_matrix_buffer(
                                &self.loaded_chunk_meshes,
                                &mut self.ring_staging_buffer,
                                self.context.clone(),
                                &self.matrix_buffer,
                                frame,
                                frame_index,
                            )?;
                        }
                        None => {}
                    },
                }
            }

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

            self.context.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            for (i, render_objs) in self.loaded_chunk_meshes.iter().enumerate() {
                for render_object in render_objs.1 {
                    let push_constants = GPUDrawPushConstants {
                        vertex_buffer: render_object.vertex_buffer_address,
                        matrix_buffer: self.matrix_buffer_address,
                        material_id: render_object.material,
                        render_object_id: i as u32,
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
                        render_object.index_buffer.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );

                    self.context.device.cmd_bind_descriptor_sets(
                        frame.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[frame.descriptors],
                        &[],
                    );

                    self.context.device.cmd_draw_indexed(
                        frame.command_buffer,
                        render_object.index_count,
                        1,
                        render_object.first_index,
                        0,
                        0,
                    );
                }
            }

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
                    .destroy_buffer(&mut self.allocator, &mut frame.uniform_buffer);
                self.context
                    .device
                    .destroy_descriptor_set_layout(frame.descriptor_layout, None);
            });

            self.global_descriptor_allocator
                .destroy_pool(&self.context.device);

            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.context.device.destroy_pipeline(self.pipeline, None);
            for loaded_chunk_meshes in &mut self.loaded_chunk_meshes {
                for loaded_chunk_mesh in loaded_chunk_meshes.1 {
                    self.context
                        .destroy_buffer(&mut self.allocator, &mut loaded_chunk_mesh.vertex_buffer);
                    self.context
                        .destroy_buffer(&mut self.allocator, &mut loaded_chunk_mesh.index_buffer);
                }
            }
            for loaded_chunk_mesh in &mut self.chunk_mesh_pool {
                self.context
                    .destroy_buffer(&mut self.allocator, &mut loaded_chunk_mesh.vertex_buffer);
                self.context
                    .destroy_buffer(&mut self.allocator, &mut loaded_chunk_mesh.index_buffer);
            }
            self.context
                .destroy_buffer(&mut self.allocator, &mut self.matrix_buffer);
            self.context
                .destroy_buffer(&mut self.allocator, &mut self.ring_staging_buffer.buffer);
            for texture in &mut self.textures {
                Texture::destroy_texture(self.context.clone(), &mut self.allocator, texture);
            }
        }
    }
}
