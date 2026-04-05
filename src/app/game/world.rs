use std::collections::HashMap;

use glam::{IVec3, Vec3};
use noise::{
    Fbm, NoiseFn, Perlin, core::perlin::perlin_3d, permutationtable::PermutationTable,
    utils::PlaneMapBuilder,
};

use crate::app::{
    engine::renderer::{
        quad::{Face, Quad},
        vertex::Vertex,
    },
    game::{
        block::BlockType,
        registry::{self},
    },
};

pub struct World {
    chunks: HashMap<IVec3, Chunk>,
}

pub struct Chunk {
    block_ids: [u32; 16 * 16 * 16],
}

pub struct ChunkMesh {
    pub material: u32,
    pub vertices: Vec<Vertex>,
    pub chunk_pos: IVec3,
}

impl World {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn generate(&mut self) {
        let perlin = Perlin::new(1);
        for i in -2..2 as i32 {
            for j in -2..2 as i32 {
                let mut chunk = Chunk {
                    block_ids: [0; 16 * 16 * 16],
                };
                for x in 0..16 {
                    for z in 0..16 {
                        let point = (perlin.get([x as f64 / 32.0, z as f64 / 32.0]) * 16.0) as u32;

                        chunk.write_block_id(x, point, z, 1);
                    }
                }
                self.chunks.insert(IVec3 { x: i, y: 0, z: j }, chunk);
            }
        }
    }

    pub fn get_block_id(&self, x: i32, y: i32, z: i32) -> Option<u32> {
        let chunk_x = x.rem_euclid(16) as usize;
        let chunk_y = y.rem_euclid(16) as usize;
        let chunk_z = z.rem_euclid(16) as usize;
        self.get_chunk(IVec3 {
            x: x.div_euclid(16),
            y: y.div_euclid(16),
            z: z.div_euclid(16),
        })
        .map(|chunk| chunk.get_block_id(chunk_x, chunk_y, chunk_z))
    }

    pub fn get_chunk(&self, pos: IVec3) -> Option<&Chunk> {
        self.chunks.get(&pos)
    }

    pub fn get_world_meshes(&self) -> Vec<ChunkMesh> {
        let chunk_meshes = self
            .chunks
            .iter()
            .flat_map(|chunk| self.get_chunk_meshes(chunk.1, chunk.0))
            .collect::<Vec<_>>();

        chunk_meshes
    }

    fn should_render_face(&self, neighbor: &BlockType) -> bool {
        neighbor.transparent
    }

    fn add_face(
        &self,
        block_type: &BlockType,
        vertices_per_material: &mut HashMap<u32, Vec<Vertex>>,
        face: Face,
        x: u32,
        y: u32,
        z: u32,
        xn: i32,
        yn: i32,
        zn: i32,
    ) {
        if self
            .get_block_id(xn, yn, zn)
            .map(|id| self.should_render_face(registry::BLOCK_REGISTRY.get(id).unwrap()))
            .unwrap_or(false)
        {
            let material = block_type.material_per_face.get(&face).unwrap();
            let quad = Quad::get_quad(
                face,
                Vec3 {
                    x: x as f32,
                    y: y as f32,
                    z: z as f32,
                },
            );
            vertices_per_material
                .entry(*material)
                .or_insert_with(Vec::new)
                .extend(quad.vertices);
        }
    }

    pub fn get_chunk_meshes(&self, chunk: &Chunk, offset: &IVec3) -> Vec<ChunkMesh> {
        let mut chunk_meshes = Vec::new();
        let mut vertices_per_material = HashMap::new();
        for block_id in chunk.block_ids.iter().enumerate() {
            let index = block_id.0 as u32;
            let x = index % 16;
            let z = (index / 16) % 16;
            let y = ((index / 16) / 16) % 16;
            let xn = offset.x * 16 + x as i32;
            let yn = offset.y * 16 + y as i32;
            let zn = offset.z * 16 + z as i32;

            if *block_id.1 == 0 {
                continue;
            }

            let block_type = registry::BLOCK_REGISTRY.get(*block_id.1).unwrap();

            self.add_face(
                block_type,
                &mut vertices_per_material,
                Face::X_PLUS,
                x,
                y,
                z,
                xn + 1,
                yn,
                zn,
            );
            self.add_face(
                block_type,
                &mut vertices_per_material,
                Face::X_MINUS,
                x,
                y,
                z,
                xn - 1,
                yn,
                zn,
            );
            self.add_face(
                block_type,
                &mut vertices_per_material,
                Face::Y_PLUS,
                x,
                y,
                z,
                xn,
                yn + 1,
                zn,
            );
            self.add_face(
                block_type,
                &mut vertices_per_material,
                Face::Y_MINUS,
                x,
                y,
                z,
                xn,
                yn - 1,
                zn,
            );
            self.add_face(
                block_type,
                &mut vertices_per_material,
                Face::Z_PLUS,
                x,
                y,
                z,
                xn,
                yn,
                zn + 1,
            );
            self.add_face(
                block_type,
                &mut vertices_per_material,
                Face::Z_MINUS,
                x,
                y,
                z,
                xn,
                yn,
                zn - 1,
            );
        }
        for v in vertices_per_material {
            chunk_meshes.push(ChunkMesh {
                material: v.0,
                vertices: v.1,
                chunk_pos: *offset,
            });
        }
        chunk_meshes
    }
}

impl Chunk {
    pub fn get_block_id(&self, x: usize, y: usize, z: usize) -> u32 {
        if x >= 16 || y >= 16 || z >= 16 {
            return 0;
        }
        let index = x + (z * 16) + (y * 16 * 16);
        return self.block_ids[index];
    }

    pub fn write_block_id(&mut self, x: u32, y: u32, z: u32, id: u32) {
        self.block_ids[(x + (z * 16) + (y * 16 * 16)) as usize] = id;
    }
}
