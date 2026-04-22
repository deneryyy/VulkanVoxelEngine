use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use anyhow::Result;
use ash::vk;
use glam::IVec3;
use gpu_allocator::vulkan::Allocator;
use parking_lot::Mutex;
use shaderc::ShaderKind;
use std::result::Result::Ok as stdOk;

use crate::app::{
    engine::{
        renderer::{
            Frame,
            context::{AllocatedBuffer, RenderingContext},
            staging::RingStagingBuffer,
            swapchain::Swapchain,
            vertex::Vertex,
        },
        resources,
    },
    game::world::{ChunkMessage, World},
};

pub enum TerrainRenderMode {
    Vertex,
    ComputeRayTracing,
}

#[repr(C, align(16))]
struct VertexTerrainPushConstants {
    vertex_buffer: vk::DeviceAddress,
    material_id: u32,
}

pub struct AllocatedChunkMesh {
    pub index_count: u32,
    pub first_index: u32,
    pub material: u32,
    pub index_buffer: AllocatedBuffer,
    pub vertex_buffer: AllocatedBuffer,
    pub vertex_buffer_address: vk::DeviceAddress,
}

pub struct TerrainRenderer {
    pub terrain_render_mode: TerrainRenderMode,
    pub ray_traversal_pipeline: vk::Pipeline,
    pub ray_traversal_pipeline_layout: vk::PipelineLayout,
    pub context: Arc<RenderingContext>,
    pub allocator: Arc<Mutex<Allocator>>,
    pub world: Arc<World>,
    pub chunk_message_queue: VecDeque<ChunkMessage>,
    pub loaded_chunk_meshes: HashMap<IVec3, Vec<AllocatedChunkMesh>>,
    pub chunk_mesh_pool: VecDeque<AllocatedChunkMesh>,
    pub vertex_pipeline: vk::Pipeline,
    pub vertex_pipeline_layout: vk::PipelineLayout,
}

impl TerrainRenderer {
    pub fn new<'a>(
        terrain_render_mode: TerrainRenderMode,
        world: Arc<World>,
        render_distance: u32,
        texture_num: u32,
        context: Arc<RenderingContext>,
        allocator: Arc<Mutex<Allocator>>,
        swapchain: &Swapchain,
        vertex_terrain_descriptor_set_layouts: &'a [vk::DescriptorSetLayout],
        ray_traversal_descriptor_set_layouts: &'a [vk::DescriptorSetLayout],
    ) -> Result<Self> {
        let mut chunk_mesh_pool = VecDeque::new();
        for _ in 0..render_distance * render_distance * 4 * texture_num {
            // Maximum number of vertices in a chunk
            let vertex_buffer_size = (16 * 16 * 16 * std::mem::size_of::<Vertex>()) as u64;

            let vertex_buffer = context.create_buffer(
                "Vertex buffer",
                &mut allocator.lock(),
                vertex_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;

            let index_buffer_size = (6 * 16 * 16 * 16 * 3 * std::mem::size_of::<u32>()) as u64;
            let index_buffer = context.create_buffer(
                "Index buffer",
                &mut allocator.lock(),
                index_buffer_size,
                vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                gpu_allocator::MemoryLocation::GpuOnly,
            )?;

            let vertex_buffer_address = unsafe {
                context.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer),
                )
            };
            chunk_mesh_pool.push_back(AllocatedChunkMesh {
                index_count: 0,
                first_index: 0,
                material: 0,
                index_buffer,
                vertex_buffer,
                vertex_buffer_address,
            });
        }

        let vertex_pipeline_layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(vertex_terrain_descriptor_set_layouts)
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .offset(0)
                        .size(std::mem::size_of::<VertexTerrainPushConstants>() as u32)
                        .stage_flags(vk::ShaderStageFlags::VERTEX)]),
                None,
            )?
        };

        let vertex_shader = resources::load_shader_module(
            "shaders/terrain.vert",
            "terrain.vert",
            ShaderKind::Vertex,
            &context.device,
        )?;
        let fragment_shader = resources::load_shader_module(
            "shaders/terrain.frag",
            "terrain.frag",
            ShaderKind::Fragment,
            &context.device,
        )?;
        let vertex_pipeline = context.create_graphics_pipeline(
            vk::Extent2D::default()
                .width(swapchain.draw_image.extent.width)
                .height(swapchain.draw_image.extent.height),
            swapchain.draw_image.image_format,
            swapchain.depth_image.image_format,
            vertex_shader,
            fragment_shader,
            vertex_pipeline_layout,
            Default::default(),
        )?;

        let ray_traversal_compute_shader = resources::load_shader_module(
            "shaders/terrain_traverse.comp",
            "terrain_traverse.comp",
            ShaderKind::Compute,
            &context.device,
        )?;

        let ray_traversal_pipeline_layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(ray_traversal_descriptor_set_layouts),
                None,
            )?
        };
        let ray_traversal_pipeline = unsafe {
            *context
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .layout(ray_traversal_pipeline_layout)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .module(ray_traversal_compute_shader)
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .name(c"main"),
                        )],
                    None,
                )
                .unwrap()
                .iter()
                .next()
                .unwrap()
        };

        unsafe {
            context.device.destroy_shader_module(vertex_shader, None);
            context.device.destroy_shader_module(fragment_shader, None);
            context
                .device
                .destroy_shader_module(ray_traversal_compute_shader, None);
        }
        Ok(Self {
            ray_traversal_pipeline,
            ray_traversal_pipeline_layout,
            terrain_render_mode,
            context,
            allocator,
            world,
            chunk_message_queue: VecDeque::new(),
            loaded_chunk_meshes: HashMap::new(),
            chunk_mesh_pool,
            vertex_pipeline,
            vertex_pipeline_layout,
        })
    }

    pub fn receive_chunks(
        &mut self,
        ring_staging_buffer: &mut RingStagingBuffer,
        frame: &Frame,
        frame_index: usize,
    ) -> Result<()> {
        while let stdOk(chunk_message) = self.world.get_receiver().try_recv() {
            self.chunk_message_queue.push_back(chunk_message);
        }

        'outer: while !self.chunk_message_queue.is_empty() {
            let chunk_message = self.chunk_message_queue.pop_front().unwrap();
            match &chunk_message {
                ChunkMessage::Load(chunk_meshes) => {
                    let mut loaded_chunks = HashMap::new();
                    for chunk_mesh in &chunk_meshes.1 {
                        if let Some(mut allocated_chunk_mesh) = self.chunk_mesh_pool.pop_front() {
                            let verts = &chunk_mesh.vertices;
                            let indices = Vertex::generate_indices(verts.len());
                            let vert_allocation = ring_staging_buffer.push(&verts, frame_index);
                            let index_allocation = ring_staging_buffer.push(&indices, frame_index);

                            if vert_allocation.is_err() || index_allocation.is_err() {
                                self.chunk_mesh_pool.push_back(allocated_chunk_mesh);
                                self.chunk_message_queue.push_back(chunk_message);
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
                                break 'outer;
                            }

                            ring_staging_buffer.transfer(
                                self.context.clone(),
                                frame.command_buffer,
                                vert_allocation?,
                                &allocated_chunk_mesh.vertex_buffer,
                                vk::PipelineStageFlags2::VERTEX_SHADER,
                                vk::AccessFlags2::SHADER_READ,
                            );
                            ring_staging_buffer.transfer(
                                self.context.clone(),
                                frame.command_buffer,
                                index_allocation?,
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
                }
                ChunkMessage::Unload(pos) => match self.loaded_chunk_meshes.remove(&pos) {
                    Some(allocated_chunk_meshes) => {
                        for allocated_chunk_mesh in allocated_chunk_meshes {
                            self.chunk_mesh_pool.push_back(allocated_chunk_mesh);
                        }
                    }
                    None => {}
                },
            }
        }

        Ok(())
    }

    pub fn record_commands(&self, frame: &Frame, extent: vk::Extent2D, swapchain: &Swapchain) {
        unsafe {
            match self.terrain_render_mode {
                TerrainRenderMode::Vertex => {
                    self.context.device.cmd_bind_pipeline(
                        frame.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.vertex_pipeline,
                    );

                    self.context.device.cmd_bind_descriptor_sets(
                        frame.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.vertex_pipeline_layout,
                        0,
                        &[frame.vertex_terrain_descriptors],
                        &[],
                    );

                    for (_, allocated_meshes) in self.loaded_chunk_meshes.iter() {
                        for allocated_mesh in allocated_meshes {
                            let push_constants = VertexTerrainPushConstants {
                                vertex_buffer: allocated_mesh.vertex_buffer_address,
                                material_id: allocated_mesh.material,
                            };
                            self.context.device.cmd_push_constants(
                                frame.command_buffer,
                                self.vertex_pipeline_layout,
                                vk::ShaderStageFlags::VERTEX,
                                0,
                                std::slice::from_raw_parts(
                                    &push_constants as *const _ as *const u8,
                                    std::mem::size_of::<VertexTerrainPushConstants>(),
                                ),
                            );

                            self.context.device.cmd_bind_index_buffer(
                                frame.command_buffer,
                                allocated_mesh.index_buffer.buffer,
                                0,
                                vk::IndexType::UINT32,
                            );

                            self.context.device.cmd_draw_indexed(
                                frame.command_buffer,
                                allocated_mesh.index_count,
                                1,
                                allocated_mesh.first_index,
                                0,
                                0,
                            );
                        }
                    }
                }
                TerrainRenderMode::ComputeRayTracing => {
                    self.context.device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_binding(0)
                            .dst_set(frame.ray_traversal_descriptors)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&[vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::GENERAL)
                                .image_view(swapchain.draw_image.image_view)])],
                        &[],
                    );
                    self.context.device.cmd_bind_pipeline(
                        frame.command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        self.ray_traversal_pipeline,
                    );
                    self.context.device.cmd_bind_descriptor_sets(
                        frame.command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        self.ray_traversal_pipeline_layout,
                        0,
                        &[frame.ray_traversal_descriptors],
                        &[],
                    );
                    self.context.device.cmd_dispatch(
                        frame.command_buffer,
                        extent.width.div_ceil(16),
                        extent.height.div_ceil(16),
                        1,
                    );
                }
            }
        }
    }
}

impl Drop for TerrainRenderer {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_pipeline_layout(self.vertex_pipeline_layout, None);
            self.context
                .device
                .destroy_pipeline(self.vertex_pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.ray_traversal_pipeline_layout, None);
            self.context
                .device
                .destroy_pipeline(self.ray_traversal_pipeline, None);
            for loaded_chunk_meshes in &mut self.loaded_chunk_meshes {
                for loaded_chunk_mesh in loaded_chunk_meshes.1 {
                    self.context.destroy_buffer(
                        &mut self.allocator.lock(),
                        &mut loaded_chunk_mesh.vertex_buffer,
                    );
                    self.context.destroy_buffer(
                        &mut self.allocator.lock(),
                        &mut loaded_chunk_mesh.index_buffer,
                    );
                }
            }
            for loaded_chunk_mesh in &mut self.chunk_mesh_pool {
                self.context.destroy_buffer(
                    &mut self.allocator.lock(),
                    &mut loaded_chunk_mesh.vertex_buffer,
                );
                self.context.destroy_buffer(
                    &mut self.allocator.lock(),
                    &mut loaded_chunk_mesh.index_buffer,
                );
            }
        }
    }
}
