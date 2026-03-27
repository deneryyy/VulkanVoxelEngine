use core::ffi;
use std::{collections::HashSet, sync::Arc};

use anyhow::{Context, Ok, Result};
use ash::vk::{self, BufferCreateInfo, ImageSubresourceLayers, ImageSubresourceRange};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

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
                    let features = instance.get_physical_device_features(handle);
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

            let device = instance.create_device(
                physical_device.handle,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_create_infos)
                    .enabled_extension_names(&[ash::khr::swapchain::NAME.as_ptr()])
                    .push_next(
                        &mut vk::PhysicalDeviceVulkan13Features::default()
                            .dynamic_rendering(true)
                            .synchronization2(true),
                    )
                    .push_next(
                        &mut vk::PhysicalDeviceVulkan12Features::default()
                            .buffer_device_address(true)
                            .descriptor_indexing(true),
                    ),
                None,
            )?;

            let swapchain_extension = ash::khr::swapchain::Device::new(&instance, &device);

            let queues = queue_family_indices
                .iter()
                .map(|index| device.get_device_queue(*index, 0))
                .collect::<Vec<_>>();

            Ok(Self {
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
        old_state: vk::ImageLayout,
        new_state: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .dst_access_mask(
                            vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
                        )
                        .old_layout(old_state)
                        .new_layout(new_state)
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(aspect_mask)
                                .base_mip_level(0)
                                .layer_count(vk::REMAINING_ARRAY_LAYERS)
                                .base_array_layer(0)
                                .level_count(vk::REMAINING_MIP_LEVELS),
                        ),
                ]),
            );
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
            self.device.cmd_blit_image2(
                command_buffer,
                &vk::BlitImageInfo2::default()
                    .dst_image(destination)
                    .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_image(source)
                    .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .filter(vk::Filter::LINEAR)
                    .regions(&[vk::ImageBlit2::default()
                        .src_offsets([
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: src_size.width as i32,
                                y: src_size.height as i32,
                                z: 1,
                            },
                        ])
                        .dst_offsets([
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: dst_size.width as i32,
                                y: dst_size.height as i32,
                                z: 1,
                            },
                        ])
                        .src_subresource(
                            ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .dst_subresource(
                            ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )]),
            );
        }
    }

    pub fn create_graphics_pipeline(
        &self,
        swapchain_extent: vk::Extent2D,
        swapchain_format: vk::Format,
        vertex_shader: vk::ShaderModule,
        fragment_shader: vk::ShaderModule,
        pipeline_layout: vk::PipelineLayout,
        pipeline_cache: vk::PipelineCache,
    ) -> Result<vk::Pipeline> {
        let entry_point = std::ffi::CString::new("main")?;

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
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
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
                                .cull_mode(vk::CullModeFlags::NONE)
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
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(&[swapchain_format]),
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
        view: vk::ImageView,
        clear_color: vk::ClearColorValue,
        render_area: vk::Rect2D,
    ) {
        unsafe {
            self.device.cmd_begin_rendering(
                command_buffer,
                &vk::RenderingInfo::default()
                    .layer_count(1)
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .clear_value(vk::ClearValue { color: clear_color })
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)])
                    .render_area(render_area),
            );
        }
    }

    pub fn create_buffer(
        &self,
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
                name: "Buffer",
                requirements,
                location,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

            self.device.bind_buffer_memory(buffer, allocation.memory(), 0)?;
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
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
