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
use tracing::info;

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
    game::{
        self,
        block::MaterialPerFace,
        registry,
        settings::TerrainRenderMode,
        world::{ChunkMessage, World},
    },
};

#[repr(C, align(16))]
struct VertexTerrainPushConstants {
    vertex_buffer: vk::DeviceAddress,
    material_id: u32,
}

#[repr(C, align(16))]
struct LegacyVertexTerrainPushConstants {
    material_id: u32,
}

#[repr(C, align(16))]
struct RayTraversalPushConstants {
    material_buffer: vk::DeviceAddress,
}

pub struct AllocatedChunkMesh {
    pub index_count: u32,
    pub first_index: u32,
    pub material: u32,
    pub index_buffer: AllocatedBuffer,
    pub vertex_buffer: AllocatedBuffer,
    pub vertex_buffer_address: Option<vk::DeviceAddress>,
}

pub struct TerrainRenderer {
    pub terrain_render_mode: TerrainRenderMode,
    pub material_buffer: AllocatedBuffer,
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
        world: Arc<World>,
        render_distance: u32,
        texture_num: u32,
        context: Arc<RenderingContext>,
        allocator: Arc<Mutex<Allocator>>,
        swapchain: &Swapchain,
        vertex_terrain_descriptor_set_layouts: &'a [vk::DescriptorSetLayout],
        texture_descriptor_set_layout: &'a vk::DescriptorSetLayout,
        ray_traversal_descriptor_set_layouts: &'a [vk::DescriptorSetLayout],
        material_buffer: AllocatedBuffer,
    ) -> Result<Self> {
        let terrain_render_mode = game::settings::GAME_SETTINGS.terrain_render_mode;
        info!("Terrain render mode: {:?}", terrain_render_mode);

        let mut chunk_mesh_pool = VecDeque::new();
        if terrain_render_mode == TerrainRenderMode::Vertex {
            for _ in 0..render_distance * render_distance * 8 * texture_num {
                // Maximum number of vertices in a chunk
                let vertex_buffer_size = (16 * 16 * 16 * std::mem::size_of::<Vertex>()) as u64;

                let vertex_usage = if context.use_buffer_device_address {
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                } else {
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER
                };
                let vertex_buffer = context.create_buffer(
                    "Vertex buffer",
                    &mut allocator.lock(),
                    vertex_buffer_size,
                    vertex_usage,
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
                    if context.use_buffer_device_address {
                        Some(context.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer),
                        ))
                    } else {
                        None
                    }
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
        }

        let mut vertex_set_layouts = Vec::new();
        vertex_set_layouts.extend_from_slice(vertex_terrain_descriptor_set_layouts);
        vertex_set_layouts.push(*texture_descriptor_set_layout);
        let vertex_push_constant_range = if context.use_buffer_device_address {
            vk::PushConstantRange::default()
                .offset(0)
                .size(std::mem::size_of::<VertexTerrainPushConstants>() as u32)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
        } else {
            vk::PushConstantRange::default()
                .offset(0)
                .size(std::mem::size_of::<LegacyVertexTerrainPushConstants>() as u32)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
        };
        let vertex_pipeline_layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&vertex_set_layouts)
                    .push_constant_ranges(&[vertex_push_constant_range]),
                None,
            )?
        };
        let bda_vertex_terrain_shader =
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/terrain.vert"));
        let legacy_vertex_terrain_shader =
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/terrain_legacy.vert"));

        let vertex_shader = if context.use_buffer_device_address {
            resources::load_shader_module(
                bda_vertex_terrain_shader,
                "terrain.vert",
                ShaderKind::Vertex,
                &context.device,
            )?
        } else {
            resources::load_shader_module(
                legacy_vertex_terrain_shader,
                "terrain_legacy.vert",
                ShaderKind::Vertex,
                &context.device,
            )?
        };
        let fragment_shader = resources::load_shader_module(
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/terrain.frag")),
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
            swapchain.main_render_pass,
        )?;

        let ray_traversal_compute_shader = resources::load_shader_module(
            include_str!(concat!(
                env!("OUT_DIR"),
                "/res/shaders/terrain_traverse.comp"
            )),
            "terrain_traverse.comp",
            ShaderKind::Compute,
            &context.device,
        )?;

        let mut ray_traversal_descriptors = Vec::new();
        ray_traversal_descriptors.extend_from_slice(ray_traversal_descriptor_set_layouts);
        ray_traversal_descriptors.push(*texture_descriptor_set_layout);
        let ray_traversal_pipeline_layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&ray_traversal_descriptors),
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
            material_buffer,
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

    pub fn create_material_buffer(
        context: Arc<RenderingContext>,
        allocator: Arc<Mutex<Allocator>>,
    ) -> Result<AllocatedBuffer> {
        let materials = registry::BLOCK_REGISTRY
            .map
            .iter()
            .map(|block| block.1.get_material_per_face())
            .collect::<Vec<MaterialPerFace>>();
        let materials_size = (std::mem::size_of::<MaterialPerFace>() * materials.len()) as u64;
        let material_buffer = context.create_buffer(
            "Material buffer",
            &mut allocator.lock(),
            materials_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
        )?;

        let mut staging_buffer = context.create_buffer(
            "Staging buffer",
            &mut allocator.lock(),
            materials_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
        )?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                materials.as_ptr(),
                staging_buffer
                    .allocation
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr()
                    .cast(),
                materials.len(),
            );
        }
        let cmd = context.begin_immediate_submit()?;
        unsafe {
            context.device.cmd_copy_buffer(
                cmd,
                staging_buffer.buffer,
                material_buffer.buffer,
                &[vk::BufferCopy::default()
                    .size(materials_size)
                    .src_offset(0)
                    .dst_offset(0)],
            );
        }
        context.end_immediate_submit()?;
        context.destroy_buffer(&mut allocator.lock(), &mut staging_buffer);
        return Ok(material_buffer);
    }

    pub fn receive_chunks(
        &mut self,
        ring_staging_buffer: &mut RingStagingBuffer,
        frame: &Frame,
        frame_index: usize,
    ) -> Result<()> {
        if self.terrain_render_mode == TerrainRenderMode::Vertex {
            while let stdOk(chunk_message) = self.world.get_receiver().try_recv() {
                self.chunk_message_queue.push_back(chunk_message);
            }
            'outer: while !self.chunk_message_queue.is_empty() {
                let chunk_message = self.chunk_message_queue.pop_front().unwrap();
                match &chunk_message {
                    ChunkMessage::VertexLoad(chunk_meshes) => {
                        let mut loaded_chunks = HashMap::new();
                        for chunk_mesh in &chunk_meshes.1 {
                            if let Some(mut allocated_chunk_mesh) = self.chunk_mesh_pool.pop_front()
                            {
                                let verts = &chunk_mesh.vertices;
                                let indices = Vertex::generate_indices(verts.len());
                                let vert_allocation = ring_staging_buffer.push(&verts, frame_index);
                                let index_allocation =
                                    ring_staging_buffer.push(&indices, frame_index);

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

                                if self.context.use_buffer_device_address {
                                    ring_staging_buffer.transfer(
                                        self.context.clone(),
                                        frame.command_buffer,
                                        vert_allocation?,
                                        &allocated_chunk_mesh.vertex_buffer,
                                        vk::PipelineStageFlags::VERTEX_SHADER,
                                        vk::AccessFlags::SHADER_READ,
                                    );
                                } else {
                                    ring_staging_buffer.transfer(
                                        self.context.clone(),
                                        frame.command_buffer,
                                        vert_allocation?,
                                        &allocated_chunk_mesh.vertex_buffer,
                                        vk::PipelineStageFlags::VERTEX_INPUT,
                                        vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                                    );
                                }
                                ring_staging_buffer.transfer(
                                    self.context.clone(),
                                    frame.command_buffer,
                                    index_allocation?,
                                    &allocated_chunk_mesh.index_buffer,
                                    vk::PipelineStageFlags::VERTEX_INPUT,
                                    vk::AccessFlags::INDEX_READ,
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
        }

        Ok(())
    }

    pub fn record_commands(
        &self,
        frame: &Frame,
        frames: &Vec<Frame>,
        extent: vk::Extent2D,
        swapchain: &Swapchain,
        texture_set: vk::DescriptorSet,
    ) {
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

                    self.context.device.cmd_bind_descriptor_sets(
                        frame.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.vertex_pipeline_layout,
                        2,
                        &[texture_set],
                        &[],
                    );

                    for (_, allocated_meshes) in self.loaded_chunk_meshes.iter() {
                        for allocated_mesh in allocated_meshes {
                            if self.context.use_buffer_device_address {
                                let push_constants = VertexTerrainPushConstants {
                                    vertex_buffer: allocated_mesh.vertex_buffer_address.unwrap(),
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
                            } else {
                                let legacy_push_constants = LegacyVertexTerrainPushConstants {
                                    material_id: allocated_mesh.material,
                                };

                                self.context.device.cmd_push_constants(
                                    frame.command_buffer,
                                    self.vertex_pipeline_layout,
                                    vk::ShaderStageFlags::VERTEX,
                                    0,
                                    std::slice::from_raw_parts(
                                        &legacy_push_constants as *const _ as *const u8,
                                        std::mem::size_of::<LegacyVertexTerrainPushConstants>(),
                                    ),
                                );

                                self.context.device.cmd_bind_vertex_buffers(
                                    frame.command_buffer,
                                    0,
                                    &[allocated_mesh.vertex_buffer.buffer],
                                    &[0],
                                );
                            }

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

                    self.context.device.cmd_bind_descriptor_sets(
                        frame.command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        self.ray_traversal_pipeline_layout,
                        2,
                        &[texture_set],
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
            self.context
                .destroy_buffer(&mut self.allocator.lock(), &mut self.material_buffer);
        }
    }
}
