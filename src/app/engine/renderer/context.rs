use std::{collections::HashSet, sync::Arc};

use anyhow::{Context, Ok, Result};
use ash::vk::{
    self, BufferCreateInfo, CommandBufferUsageFlags, ImageSubresourceLayers,
    PipelineDepthStencilStateCreateInfo, RenderingAttachmentInfo,
};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
};
use tracing::info;
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

use crate::app::engine::renderer::vertex::Vertex;

#[derive(Clone)]
pub struct QueueFamily {
    pub index: u32,
    pub properties: vk::QueueFamilyProperties,
}

pub struct QueueFamilies {
    pub graphics: u32,
    pub present: u32,
    pub transfer: u32,
    pub compute: u32,
}

pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_families: Vec<QueueFamily>,
}

pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

#[derive(Default)]
pub struct AllocatedImage {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub allocation: Allocation,
    pub extent: vk::Extent3D,
    pub image_format: vk::Format,
}

pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
}

pub struct RenderingContext {
    pub use_buffer_device_address: bool,
    pub use_dynamic_rendering: bool,
    pub use_sync2: bool,
    imm_fence: vk::Fence,
    imm_command_buffer: vk::CommandBuffer,
    imm_command_pool: vk::CommandPool,
    pub queues: Vec<vk::Queue>,
    pub swapchain_extension: ash::khr::swapchain::Device,
    pub device: ash::Device,
    pub queue_family_indices: HashSet<u32>,
    pub queue_families: QueueFamilies,
    pub physical_device: PhysicalDevice,
    pub surface_extension: ash::khr::surface::Instance,
    pub instance: ash::Instance,
    pub entry: ash::Entry,
}

impl RenderingContext {
    fn pick_queue_families(
        physical_devices: Vec<PhysicalDevice>,
    ) -> Result<(PhysicalDevice, QueueFamilies)> {
        let physical_device = physical_devices.into_iter().next().unwrap();
        let queue_family = physical_device
            .queue_families
            .iter()
            .find(|queue_family| {
                queue_family
                    .properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
                    && queue_family
                        .properties
                        .queue_flags
                        .contains(vk::QueueFlags::COMPUTE)
            })
            .map(|queue_family| queue_family.index)
            .context("No suitable queue family found")?;

        Ok((
            physical_device,
            QueueFamilies {
                graphics: queue_family,
                present: queue_family,
                transfer: queue_family,
                compute: queue_family,
            },
        ))
    }

    pub fn new(window: Arc<Window>) -> Result<Self> {
        unsafe {
            let raw_display_handle = window.display_handle()?.as_raw();
            let raw_window_handle = window.window_handle()?.as_raw();

            let entry = ash::Entry::load()?;
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(
                        &vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3),
                    )
                    .enabled_extension_names(ash_window::enumerate_required_extensions(
                        raw_display_handle,
                    )?),
                None,
            )?;

            let surface_extension = ash::khr::surface::Instance::new(&entry, &instance);

            let compatibility_surface = ash_window::create_surface(
                &entry,
                &instance,
                raw_display_handle,
                raw_window_handle,
                None,
            )?;

            let mut physical_devices = instance
                .enumerate_physical_devices()?
                .into_iter()
                .map(|handle| {
                    let properties = instance.get_physical_device_properties(handle);
                    let features = instance
                        .get_physical_device_features(handle)
                        .sampler_anisotropy(true);
                    let memory_properties = instance.get_physical_device_memory_properties(handle);
                    let queue_family_properties =
                        instance.get_physical_device_queue_family_properties(handle);

                    let queue_families = queue_family_properties
                        .into_iter()
                        .enumerate()
                        .map(|(index, properties)| QueueFamily {
                            index: index as u32,
                            properties,
                        })
                        .collect::<Vec<_>>();

                    PhysicalDevice {
                        handle,
                        properties,
                        features,
                        memory_properties,
                        queue_families,
                    }
                })
                .collect::<Vec<_>>();

            physical_devices.retain(|device| {
                surface_extension
                    .get_physical_device_surface_support(device.handle, 0, compatibility_surface)
                    .unwrap_or(false)
            });

            surface_extension.destroy_surface(compatibility_surface, None);

            let (physical_device, queue_families) = Self::pick_queue_families(physical_devices)?;

            let queue_family_indices: HashSet<u32> = HashSet::from([
                queue_families.graphics,
                queue_families.present,
                queue_families.transfer,
                queue_families.compute,
            ]);

            let queue_create_infos = queue_family_indices
                .iter()
                .copied()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(index)
                        .queue_priorities(&[1.0])
                })
                .collect::<Vec<_>>();

            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut v13)
                .push_next(&mut v12);
            instance.get_physical_device_features2(physical_device.handle, &mut f2);

            let device_api = physical_device.properties.api_version;
            let use_sync2 = device_api >= vk::API_VERSION_1_3 && v13.synchronization2 != 0;
            let use_dynamic_rendering =
                device_api >= vk::API_VERSION_1_3 && v13.dynamic_rendering != 0;
            let use_buffer_device_address =
                device_api >= vk::API_VERSION_1_2 && v12.buffer_device_address != 0;
            assert!(
                device_api > vk::API_VERSION_1_2,
                "This engine supports only vulkan 1.2 or upwards!"
            );
            info!("using sync2: {}", use_sync2);
            info!("using dynamic rendering: {}", use_dynamic_rendering);
            info!("using buffer device address: {}", use_buffer_device_address);

            let extension_names = Vec::from([ash::khr::swapchain::NAME.as_ptr()]);
            let mut v13_features = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);
            let mut v12_features = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(use_buffer_device_address)
                .descriptor_indexing(true)
                .descriptor_binding_storage_image_update_after_bind(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_variable_descriptor_count(true)
                .descriptor_binding_partially_bound(true)
                .runtime_descriptor_array(true)
                .shader_sampled_image_array_non_uniform_indexing(true);
            let mut device_info = vk::DeviceCreateInfo::default()
                .enabled_features(&physical_device.features)
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&extension_names);
            if device_api >= vk::API_VERSION_1_3 {
                device_info = device_info.push_next(&mut v13_features);
            }
            device_info = device_info.push_next(&mut v12_features);
            let device = instance.create_device(physical_device.handle, &device_info, None)?;

            let swapchain_extension = ash::khr::swapchain::Device::new(&instance, &device);

            let queues = queue_family_indices
                .iter()
                .map(|index| device.get_device_queue(*index, 0))
                .collect::<Vec<_>>();

            let imm_command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_families.graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;

            let &imm_command_buffer = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(imm_command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?
                .iter()
                .next()
                .unwrap();

            let imm_fence = device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;

            Ok(Self {
                use_buffer_device_address,
                use_dynamic_rendering,
                use_sync2,
                imm_fence,
                imm_command_buffer,
                imm_command_pool,
                queues,
                swapchain_extension,
                device,
                queue_family_indices,
                queue_families,
                physical_device,
                surface_extension,
                instance,
                entry,
            })
        }
    }

    pub fn create_surface(&self, window: &Window) -> Result<Surface> {
        unsafe {
            let raw_display_handle = window.display_handle()?.as_raw();
            let raw_window_handle = window.window_handle()?.as_raw();

            let handle = ash_window::create_surface(
                &self.entry,
                &self.instance,
                raw_display_handle,
                raw_window_handle,
                None,
            )?;

            let capabilities = self
                .surface_extension
                .get_physical_device_surface_capabilities(self.physical_device.handle, handle)?;
            let formats = self
                .surface_extension
                .get_physical_device_surface_formats(self.physical_device.handle, handle)?;
            let present_modes = self
                .surface_extension
                .get_physical_device_surface_present_modes(self.physical_device.handle, handle)?;

            Ok(Surface {
                handle,
                capabilities,
                formats,
                present_modes,
            })
        }
    }

    pub fn create_image(
        &self,
        allocator: &mut Allocator,
        extent: vk::Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_flags: vk::ImageAspectFlags,
    ) -> Result<AllocatedImage> {
        let image_handle = unsafe {
            self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(extent)
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?
        };

        let requirements = unsafe { self.device.get_image_memory_requirements(image_handle) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "Image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            self.device
                .bind_image_memory(image_handle, allocation.memory(), allocation.offset())?
        };

        let image_view = self.create_image_view(image_handle, format, aspect_flags)?;

        Ok(AllocatedImage {
            image: image_handle,
            image_view,
            allocation: allocation,
            image_format: format,
            extent,
        })
    }

    pub fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView> {
        let image_view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(aspect_flags)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
                None,
            )
        }?;

        Ok(image_view)
    }

    pub fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
        src_stage: vk::PipelineStageFlags, // Keep passing legacy flags for the 1.2 path
        dst_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) {
        unsafe {
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_mask)
                .base_mip_level(0)
                .level_count(vk::REMAINING_MIP_LEVELS)
                .base_array_layer(0)
                .layer_count(vk::REMAINING_ARRAY_LAYERS);

            if self.use_sync2 {
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::from_raw(src_stage.as_raw() as u64))
                    .src_access_mask(vk::AccessFlags2::from_raw(src_access.as_raw() as u64))
                    .dst_stage_mask(vk::PipelineStageFlags2::from_raw(dst_stage.as_raw() as u64))
                    .dst_access_mask(vk::AccessFlags2::from_raw(dst_access.as_raw() as u64))
                    .old_layout(old_layout)
                    .new_layout(new_layout)
                    .image(image)
                    .subresource_range(range);

                self.device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
                );
            } else {
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(src_access)
                    .dst_access_mask(dst_access)
                    .old_layout(old_layout)
                    .new_layout(new_layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range);

                self.device.cmd_pipeline_barrier(
                    command_buffer,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }
    }

    pub fn copy_image_to_image(
        &self,
        command_buffer: vk::CommandBuffer,
        source: vk::Image,
        destination: vk::Image,
        src_size: vk::Extent2D,
        dst_size: vk::Extent2D,
    ) {
        unsafe {
            self.device.cmd_blit_image(
                command_buffer,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .base_array_layer(0)
                            .mip_level(0),
                    )
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: src_size.width as i32,
                            y: src_size.height as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .base_array_layer(0)
                            .mip_level(0),
                    )
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: dst_size.width as i32,
                            y: dst_size.height as i32,
                            z: 1,
                        },
                    ])],
                vk::Filter::LINEAR,
            );
        }
    }

    pub fn begin_immediate_submit(&self) -> Result<vk::CommandBuffer> {
        unsafe {
            self.device.reset_fences(&[self.imm_fence])?;
            self.device.reset_command_buffer(
                self.imm_command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )?;
            self.device.begin_command_buffer(
                self.imm_command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            Ok(self.imm_command_buffer)
        }
    }

    pub fn end_immediate_submit(&self) -> Result<()> {
        unsafe {
            self.device.end_command_buffer(self.imm_command_buffer)?;

            if self.use_sync2 {
                self.device.queue_submit2(
                    self.queues[self.queue_families.graphics as usize],
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default()
                            .command_buffer(self.imm_command_buffer)])],
                    self.imm_fence,
                )?;
            } else {
                let submit_info = vk::SubmitInfo::default()
                    .command_buffers(std::slice::from_ref(&self.imm_command_buffer));

                self.device.queue_submit(
                    self.queues[self.queue_families.graphics as usize],
                    &[submit_info],
                    self.imm_fence,
                )?;
            }
            self.device
                .wait_for_fences(&[self.imm_fence], true, u64::MAX)?;
            Ok(())
        }
    }

    pub fn create_graphics_pipeline(
        &self,
        swapchain_extent: vk::Extent2D,
        swapchain_format: vk::Format,
        depth_format: vk::Format,
        vertex_shader: vk::ShaderModule,
        fragment_shader: vk::ShaderModule,
        pipeline_layout: vk::PipelineLayout,
        pipeline_cache: vk::PipelineCache,
        render_pass: Option<vk::RenderPass>,
    ) -> Result<vk::Pipeline> {
        let entry_point = std::ffi::CString::new("main")?;

        let binding_descriptions = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];

        let attribute_descriptions = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, pos) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, uv1) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, normal) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(3)
                .format(vk::Format::R32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, uv2) as u32),
        ];

        let vertex_input_state = if self.use_buffer_device_address {
            vk::PipelineVertexInputStateCreateInfo::default()
        } else {
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&binding_descriptions)
                .vertex_attribute_descriptions(&attribute_descriptions)
        };

        unsafe {
            Ok(self
                .device
                .create_graphics_pipelines(
                    pipeline_cache,
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .module(vertex_shader)
                                .name(&entry_point),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .module(fragment_shader)
                                .name(&entry_point),
                        ])
                        .vertex_input_state(&vertex_input_state)
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewports(&[vk::Viewport::default()
                                    .width(swapchain_extent.width as f32)
                                    .height(swapchain_extent.height as f32)
                                    .max_depth(1.0)])
                                .scissors(&[vk::Rect2D::default().extent(swapchain_extent)]),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .polygon_mode(vk::PolygonMode::FILL)
                                .cull_mode(vk::CullModeFlags::BACK)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
                                vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                                    .blend_enable(false),
                            ]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(pipeline_layout)
                        .depth_stencil_state(
                            &PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                                .depth_bounds_test_enable(false)
                                .stencil_test_enable(false)
                                .min_depth_bounds(0.0)
                                .max_depth_bounds(1.0),
                        )
                        .render_pass(render_pass.unwrap_or(vk::RenderPass::null()))
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(&[swapchain_format])
                                .depth_attachment_format(depth_format),
                        )],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap())
        }
    }

    pub fn begin_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        color_view: vk::ImageView,
        depth_view: vk::ImageView,
        clear_color: vk::ClearColorValue,
        render_area: vk::Rect2D,
        framebuffer: Option<vk::Framebuffer>,
        main_render_pass: Option<vk::RenderPass>,
    ) {
        unsafe {
            if self.use_dynamic_rendering {
                self.device.cmd_begin_rendering(
                    command_buffer,
                    &vk::RenderingInfo::default()
                        .layer_count(1)
                        .color_attachments(&[vk::RenderingAttachmentInfo::default()
                            .image_view(color_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .clear_value(vk::ClearValue { color: clear_color })
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)])
                        .depth_attachment(
                            &RenderingAttachmentInfo::default()
                                .image_view(depth_view)
                                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                .clear_value(vk::ClearValue {
                                    depth_stencil: vk::ClearDepthStencilValue {
                                        depth: 1.0,
                                        stencil: 0,
                                    },
                                })
                                .load_op(vk::AttachmentLoadOp::CLEAR)
                                .store_op(vk::AttachmentStoreOp::STORE),
                        )
                        .render_area(render_area),
                );
            } else {
                let clear_values = [
                    vk::ClearValue { color: clear_color },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ];
                let begin = vk::RenderPassBeginInfo::default()
                    .render_pass(main_render_pass.unwrap())
                    .framebuffer(framebuffer.unwrap())
                    .render_area(render_area)
                    .clear_values(&clear_values);
                self.device.cmd_begin_render_pass(
                    command_buffer,
                    &begin,
                    vk::SubpassContents::INLINE,
                );
            }
        }
    }

    pub fn create_main_render_pass(
        &self,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE) // depth not read after pass
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .depth_stencil_attachment(&depth_ref);

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let attachments = [color_attachment, depth_attachment];
        let dependencies = [dependency];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        let render_pass = unsafe { self.device.create_render_pass(&render_pass_info, None)? };

        Ok(render_pass)
    }

    pub fn create_main_framebuffer(
        &self,
        color_view: vk::ImageView,
        depth_view: vk::ImageView,
        extent: vk::Extent2D,
        main_render_pass: Option<vk::RenderPass>,
    ) -> Result<vk::Framebuffer> {
        let attachments = [color_view, depth_view];

        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(main_render_pass.unwrap())
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let framebuffer = unsafe { self.device.create_framebuffer(&framebuffer_info, None)? };

        Ok(framebuffer)
    }

    pub fn create_buffer(
        &self,
        name: &str,
        allocator: &mut Allocator,
        alloc_size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<AllocatedBuffer> {
        unsafe {
            let buffer = self.device.create_buffer(
                &BufferCreateInfo::default().usage(usage).size(alloc_size),
                None,
            )?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let allocation = allocator.allocate(&AllocationCreateDesc {
                name: name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            Ok(AllocatedBuffer { buffer, allocation })
        }
    }

    pub fn destroy_buffer(&self, allocator: &mut Allocator, buffer: &mut AllocatedBuffer) {
        allocator
            .free(std::mem::take(&mut buffer.allocation))
            .unwrap();
        unsafe { self.device.destroy_buffer(buffer.buffer, None) };
    }
}

impl Drop for RenderingContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.imm_fence, None);
            self.device
                .destroy_command_pool(self.imm_command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
