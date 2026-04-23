use std::{collections::HashMap, sync::OnceLock};

use crate::app::{
    engine::renderer::quad::{Face, Quad},
    game::registry::Registry,
};

pub struct BlockType {
    pub material_per_face: HashMap<Face, u32>,
    pub transparent: bool,
}

pub struct MaterialPerFace {
    x_plus: u32,
    x_minus: u32,
    y_plus: u32,
    y_minus: u32,
    z_plus: u32,
    z_minus: u32,
}

impl BlockType {
    pub fn new(material_per_face: HashMap<Face, u32>, transparent: bool) -> Self {
        Self {
            material_per_face,
            transparent,
        }
    }

    pub fn get_material_per_face(&self) -> MaterialPerFace {
        MaterialPerFace {
            x_plus: *self.material_per_face.get(&Face::XPlus).unwrap_or(&0),
            x_minus: *self.material_per_face.get(&Face::XMinus).unwrap_or(&0),
            y_plus: *self.material_per_face.get(&Face::YPlus).unwrap_or(&0),
            y_minus: *self.material_per_face.get(&Face::YMinus).unwrap_or(&0),
            z_plus: *self.material_per_face.get(&Face::ZPlus).unwrap_or(&0),
            z_minus: *self.material_per_face.get(&Face::ZMinus).unwrap_or(&0),
        }
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
