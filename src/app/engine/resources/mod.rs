use std::io::Cursor;

use byteorder::{ReadBytesExt, LittleEndian, BigEndian};

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

pub fn load_binary_u32(file_name: &str) -> anyhow::Result<Vec<u32>> {
    let data = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read(path)?
    };

    Ok(u8_to_u32_le_vec(&data))
}

fn u8_to_u32_le_vec(data: &[u8]) -> Vec<u32> {
    // Ensure the slice length is a multiple of 4 (the size of a u32)
    assert_eq!(data.len() % 4, 0, "Slice length must be a multiple of 4");

    data.chunks_exact(4) // Iterate over the slice in exact 4-byte chunks
        .map(|chunk| {
            // Convert the slice chunk into a fixed-size array [u8; 4]
            let bytes: [u8; 4] = chunk.try_into().expect("Chunk size is exactly 4");
            // Interpret the 4 bytes as a u32 in little-endian format
            u32::from_le_bytes(bytes)
        })
        .collect() // Collect the resulting u32 values into a Vec
}