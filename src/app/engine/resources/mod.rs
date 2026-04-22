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

pub fn load_spirv(data: Vec<u8>) -> anyhow::Result<Vec<u32>> {
    Ok(ash::util::read_spv(&mut std::io::Cursor::new(data))?)
}

pub fn load_shader_module(path: &str, input_file_name: &str, shader_kind: shaderc::ShaderKind, device: &ash::Device) -> anyhow::Result<vk::ShaderModule> {
    let binary_data = load_spirv(compile_shader(load_string(path)?.as_str(), input_file_name, shader_kind)?)?;

    unsafe {
        Ok(device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&binary_data),
            None,
        )?)
    }
}

fn compile_shader(source: &str, input_file_name: &str, shader_kind: shaderc::ShaderKind) -> anyhow::Result<Vec<u8>> {
    let compiler = shaderc::Compiler::new().unwrap();
    let options = shaderc::CompileOptions::new().unwrap();
    //options.add_macro_definition("EP", Some("main"));
    let binary_result = compiler
        .compile_into_spirv(
            source,
            shader_kind,
            &input_file_name,
            "main",
            Some(&options),
        )
        .unwrap();

    Ok(Vec::from(binary_result.as_binary_u8()))
}
