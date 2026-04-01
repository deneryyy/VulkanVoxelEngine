use ash::vk;

pub fn load_string(file_name: &str) -> anyhow::Result<String> {
    let txt = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read_to_string(path)?
    };

    Ok(txt)
}

pub fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    let data = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read(path)?
    };

    Ok(data)
}

pub fn load_spirv(file_name: &str) -> anyhow::Result<Vec<u32>> {
    let data = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read(path)?
    };
    Ok(ash::util::read_spv(&mut std::io::Cursor::new(data))?)
}

pub fn load_shader_module(path: &str, device: &ash::Device) -> anyhow::Result<vk::ShaderModule> {
    let binary_data = load_spirv(path)?;

    unsafe {
        Ok(device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&binary_data),
            None,
        )?)
    }
}
