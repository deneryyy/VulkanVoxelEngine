use std::{collections::HashMap, sync::OnceLock};

use crate::app::{
    engine::renderer::quad::{Face, Quad},
    game::registry::Registry,
};

pub struct BlockType {
    pub material_per_face: HashMap<Face, u32>,
    pub transparent: bool,
}

impl BlockType {
    pub fn new(material_per_face: HashMap<Face, u32>, transparent: bool) -> Self {
        Self { material_per_face, transparent }
    }

    pub fn all_faces(material_id: u32) -> HashMap<Face, u32> {
        HashMap::from([
            (Face::X_PLUS, material_id),
            (Face::X_MINUS, material_id),
            (Face::Y_PLUS, material_id),
            (Face::Y_MINUS, material_id),
            (Face::Z_PLUS, material_id),
            (Face::Z_MINUS, material_id),
        ])
    }
    pub fn side_faces(material_id: u32, material_id2: u32) -> HashMap<Face, u32> {
        HashMap::from([
            (Face::X_PLUS, material_id),
            (Face::X_MINUS, material_id),
            (Face::Y_PLUS, material_id2),
            (Face::Y_MINUS, material_id2),
            (Face::Z_PLUS, material_id),
            (Face::Z_MINUS, material_id),
        ])
    }
}
