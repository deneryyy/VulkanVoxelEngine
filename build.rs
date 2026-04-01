use anyhow::*;
use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;
use shaderc::ShaderKind;
use std::path::PathBuf;
use std::{env, fs};

fn compile_and_output_shader(src_path: &str, input_file_name: &str, output_path: &str, shader_kind: ShaderKind) -> Result<()> {
    let compute_shader_src = fs::read_to_string(src_path)?;
    let source = &compute_shader_src.as_str();
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
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
    fs::write(output_path, binary_result.as_binary_u8())?;

    Ok(())
}

fn main() -> Result<()> {
    // This tells Cargo to rerun this script if something in /res/ changes.
    println!("cargo:rerun-if-changed=res/*");

    compile_and_output_shader("res/shaders/shader.frag", "shader.frag", "res/compiled_shaders/shader.frag.spv", ShaderKind::Fragment)?;
    compile_and_output_shader("res/shaders/shader.vert", "shader.vert", "res/compiled_shaders/shader.vert.spv", ShaderKind::Vertex)?;

    let out_dir = env::var("OUT_DIR")?;
    let mut copy_options = CopyOptions::new();
    copy_options.overwrite = true;
    let mut paths_to_copy = Vec::new();
    paths_to_copy.push("res/");
    copy_items(&paths_to_copy, out_dir, &copy_options)?;

    Ok(())
}
