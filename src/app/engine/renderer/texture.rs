use std::{io::Cursor, sync::Arc};

use anyhow::{Ok, Result};
use ash::vk::{self, ImageSubresourceLayers, Offset3D};
use gpu_allocator::vulkan::Allocator;
use image::{ImageReader, Rgba};

use crate::app::engine::{
    renderer::context::{AllocatedImage, RenderingContext},
    resources,
};

pub struct Texture {
    pub image: AllocatedImage,
    pub sampler: vk::Sampler,
}

impl Texture {
    pub fn load_texture_from_file(
        path: &str,
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
    ) -> Result<Texture> {
        Self::load_texture(resources::load_binary(path)?, context, allocator)
    }

    pub fn load_texture(
        data: Vec<u8>,
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
    ) -> Result<Texture> {
        let img = ImageReader::new(Cursor::new(data))
            .with_guessed_format()?
            .decode()?;
        let buffer = img.to_rgba8();
        let mut staging = context.create_buffer(
            allocator,
            (buffer.len() * std::mem::size_of::<Rgba<f32>>()) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
        )?;
        let data = staging.allocation.mapped_ptr().unwrap().as_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(buffer.as_ptr(), data.cast(), buffer.len());
        }
        let extent = vk::Extent3D {
            width: img.width(),
            height: img.height(),
            depth: 1,
        };
        let image = context.create_image(
            allocator,
            extent,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;
        let cmd = context.begin_immediate_submit()?;
        unsafe {
            context.transition_image_layout(
                cmd,
                image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );
            context.device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_offset(Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(extent)
                    .image_subresource(
                        ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )],
            );
            context.transition_image_layout(
                cmd,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );
        }
        context.end_immediate_submit()?;
        context.destroy_buffer(allocator, &mut staging);

        let sampler = unsafe {
            let max_anisotropy = context
                .physical_device
                .properties
                .limits
                .max_sampler_anisotropy;
            context.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .anisotropy_enable(true)
                    .max_anisotropy(max_anisotropy)
                    .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                    .unnormalized_coordinates(false)
                    .compare_enable(true)
                    .compare_op(vk::CompareOp::ALWAYS)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .mip_lod_bias(0.0)
                    .min_lod(0.0)
                    .max_lod(0.0),
                None,
            )?
        };

        Ok(Self { image, sampler })
    }

    pub fn destroy_texture(
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
        texture: &mut Texture,
    ) {
        unsafe {
            context.device.destroy_sampler(texture.sampler, None);
            context.device.destroy_image(texture.image.image, None);
            context
                .device
                .destroy_image_view(texture.image.image_view, None);
            allocator
                .free(std::mem::take(&mut texture.image.allocation))
                .unwrap();
        }
    }
}
