use ash::vk;
use anyhow::Result;

pub struct DescriptorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl<'a> DescriptorLayoutBuilder<'a> {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    pub fn add_binding(&mut self, binding: u32, desc_type: vk::DescriptorType) {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_count(1)
                .descriptor_type(desc_type),
        );
    }

    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    pub fn build(
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

pub struct DescriptorAllocator {
    pool: vk::DescriptorPool,
}
pub struct PoolSizeRatio {
    pub desc_type: vk::DescriptorType,
    pub ratio: u32,
}

impl DescriptorAllocator {
    pub fn new(device: &ash::Device, max_sets: u32, pool_ratios: Vec<PoolSizeRatio>) -> Result<Self> {
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

    pub fn clear_descriptors(&self, device: &ash::Device) {
        unsafe {
            device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty());
        }
    }

    pub fn destroy_pool(&self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
        }
    }

    pub fn allocate(
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