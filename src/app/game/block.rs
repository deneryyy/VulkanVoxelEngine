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
            (Face::XPlus, material_id),
            (Face::XMinus, material_id),
            (Face::YPlus, material_id),
            (Face::YMinus, material_id),
            (Face::ZPlus, material_id),
            (Face::ZMinus, material_id),
        ])
    }
    pub fn side_faces(material_id: u32, material_id2: u32) -> HashMap<Face, u32> {
        HashMap::from([
            (Face::XPlus, material_id),
            (Face::XMinus, material_id),
            (Face::YPlus, material_id2),
            (Face::YMinus, material_id2),
            (Face::ZPlus, material_id),
            (Face::ZMinus, material_id),
        ])
    }
}
