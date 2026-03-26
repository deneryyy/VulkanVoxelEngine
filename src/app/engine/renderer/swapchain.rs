use std::sync::Arc;

use anyhow::Result;
use ash::vk::{self, ImageSubresourceRange};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use winit::window::Window;

use crate::app::engine::renderer::context::{AllocatedImage, RenderingContext, Surface};
use std::result::Result::Ok as stdOk;

pub struct Swapchain {
    allocator: Allocator,
    pub draw_image: AllocatedImage,
    pub desired_image_count: u32,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub image_views: Vec<vk::ImageView>,
    pub images: Vec<vk::Image>,
    pub handle: vk::SwapchainKHR,
    pub surface: Surface,
    pub window: Arc<Window>,
    pub context: Arc<RenderingContext>,
    pub is_dirty: bool,
}

impl Swapchain {
    pub fn new(context: Arc<RenderingContext>, window: Arc<Window>) -> Result<Self> {
        let surface = unsafe { context.create_surface(window.as_ref())? };
        let format = vk::Format::B8G8R8A8_SRGB;
        let extent = if surface.capabilities.current_extent.width != u32::MAX {
            surface.capabilities.current_extent
        } else {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width,
                height: size.height,
            }
        };
        let desired_image_count = (surface.capabilities.min_image_count + 1).clamp(
            surface.capabilities.min_image_count,
            if surface.capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                surface.capabilities.max_image_count
            },
        );

        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: context.instance.clone(),
            device: context.device.clone(),
            physical_device: context.physical_device.handle,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let draw_image = Self::create_draw_image(window.clone(), context.clone(), &mut allocator)?;

        Ok(Self {
            allocator,
            draw_image,
            desired_image_count,
            format,
            extent,
            image_views: Default::default(),
            images: Default::default(),
            handle: Default::default(),
            surface,
            window,
            context,
            is_dirty: true,
        })
    }

    pub fn resize(&mut self) -> Result<()> {
        let size = self.window.inner_size();
        self.extent = vk::Extent2D {
            width: size.width,
            height: size.height,
        };

        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(());
        }

        unsafe {
            self.context.device.device_wait_idle()?;
            let surface_capabilities = self
                .context
                .surface_extension
                .get_physical_device_surface_capabilities(
                    self.context.physical_device.handle,
                    self.surface.handle,
                )?;

            let new_swapchain = self.context.swapchain_extension.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface.handle)
                    .min_image_count(self.desired_image_count)
                    .image_format(self.format)
                    .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                    .image_extent(self.extent)
                    .image_array_layers(1)
                    .image_usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                    )
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(vk::PresentModeKHR::FIFO)
                    .clipped(true)
                    .old_swapchain(self.handle),
                None,
            )?;
            // Destroying previous image views and swapchain
            self.image_views.drain(..).for_each(|image_view| {
                self.context.device.destroy_image_view(image_view, None);
            });
            self.images.clear();
            self.context
                .swapchain_extension
                .destroy_swapchain(self.handle, None);

            // Destroying draw image
            self.context
                .device
                .destroy_image(self.draw_image.image, None);
            self.context
                .device
                .destroy_image_view(self.draw_image.image_view, None);
            self.allocator
                .free(std::mem::take(&mut self.draw_image.allocation))
                .unwrap();
            // getting new ones
            self.handle = new_swapchain;
            self.images = self
                .context
                .swapchain_extension
                .get_swapchain_images(self.handle)?;
            for image in &self.images {
                self.image_views.push(self.context.create_image_view(
                    *image,
                    self.format,
                    vk::ImageAspectFlags::COLOR,
                )?);
            }

            self.draw_image = Self::create_draw_image(
                self.window.clone(),
                self.context.clone(),
                &mut self.allocator,
            )?;
        }

        Ok(())
    }

    fn create_draw_image(
        window: Arc<Window>,
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
    ) -> Result<AllocatedImage> {
        let draw_image_extent = vk::Extent3D {
            width: window.inner_size().width,
            height: window.inner_size().height,
            depth: 1,
        };

        let draw_image_handle = unsafe {
            context.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .extent(draw_image_extent)
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?
        };

        let requirements = unsafe {
            context
                .device
                .get_image_memory_requirements(draw_image_handle)
        };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "My Image Allocation",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            context.device.bind_image_memory(
                draw_image_handle,
                allocation.memory(),
                vk::DeviceSize::default(),
            )?
        };

        let image_view = unsafe {
            context.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .image(draw_image_handle)
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .subresource_range(
                        ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .layer_count(1)
                            .base_array_layer(0)
                            .level_count(1),
                    ),
                None,
            )?
        };

        Ok(AllocatedImage {
            image: draw_image_handle,
            image_view,
            allocation: allocation,
            image_format: vk::Format::R16G16B16A16_SFLOAT,
            extent: draw_image_extent,
        })
    }

    pub fn acquire_next_image(&mut self, image_available_semaphore: vk::Semaphore) -> Result<u32> {
        let (image_index, is_suboptimal) = unsafe {
            self.context.swapchain_extension.acquire_next_image(
                self.handle,
                u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )?
        };
        if is_suboptimal {
            self.is_dirty = true;
        }
        Ok(image_index)
    }

    pub fn present(
        &mut self,
        image_index: u32,
        render_finished_semaphore: vk::Semaphore,
    ) -> Result<()> {
        let is_suboptimal = unsafe {
            match self.context.swapchain_extension.queue_present(
                self.context.queues[self.context.queue_families.present as usize],
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&[render_finished_semaphore])
                    .swapchains(&[self.handle])
                    .image_indices(&[image_index]),
            ) {
                stdOk(is_suboptimal) => is_suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
                Err(error) => return Err(error.into()),
            }
        };
        if is_suboptimal {
            self.is_dirty = true;
        }

        Ok(())
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.image_views.drain(..).for_each(|image_view| {
                self.context.device.destroy_image_view(image_view, None);
            });
            self.images.clear();
            self.context
                .swapchain_extension
                .destroy_swapchain(self.handle, None);
            self.context
                .surface_extension
                .destroy_surface(self.surface.handle, None);
            self.context
                .device
                .destroy_image(self.draw_image.image, None);
            self.context
                .device
                .destroy_image_view(self.draw_image.image_view, None);
            self.allocator
                .free(std::mem::take(&mut self.draw_image.allocation))
                .unwrap();
        }
    }
}
