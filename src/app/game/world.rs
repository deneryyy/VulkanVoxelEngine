use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crossbeam_channel::{Receiver, Sender, bounded};
use glam::{IVec3, Vec3};
use noise::{
    Fbm, MultiFractal, NoiseFn, Perlin, core::perlin::perlin_3d,
    permutationtable::PermutationTable, utils::PlaneMapBuilder,
};
use parking_lot::RwLock;
use rayon::{
    ThreadPool, ThreadPoolBuilder,
    iter::{IntoParallelIterator, ParallelIterator},
};

use crate::app::{
    engine::{
        input::InputManager,
        renderer::{
            camera::Camera,
            quad::{Face, Quad},
            vertex::Vertex,
        },
    },
    game::{
        block::BlockType,
        player::Player,
        registry::{self},
    },
};

pub struct World {
    seed: u32,
    fbm_perlin: Fbm<Perlin>,
    thread_pool: ThreadPool,
    sender: Sender<ChunkMessage>,
    receiver: Receiver<ChunkMessage>,
    chunks: RwLock<HashMap<IVec3, Chunk>>,
    pub player: RwLock<Player>,
}

pub struct Chunk {
    block_ids: [u32; 16 * 16 * 16],
}

pub struct ChunkMesh {
    pub material: u32,
    pub vertices: Vec<Vertex>,
    pub chunk_pos: IVec3,
}

pub enum ChunkMessage {
    Load((IVec3, Vec<ChunkMesh>)),
    Unload(IVec3),
}

impl World {
    pub fn new(seed: u32, input_manager: Arc<InputManager>) -> Self {
        let (sender, receiver) = bounded(64);
        Self {
            seed,
            fbm_perlin: Fbm::<Perlin>::new(seed)
                .set_octaves(4)
                .set_frequency(0.02)
                .set_lacunarity(2.0)
                .set_persistence(0.5),
            thread_pool: ThreadPoolBuilder::new().num_threads(8).build().unwrap(),
            sender,
            receiver,
            chunks: RwLock::new(HashMap::new()),
            player: RwLock::new(Player::new(
                input_manager.clone(),
                Vec3 {
                    x: 2.0,
                    y: 200.0,
                    z: 2.0,
                },
                Camera::new(
                    Vec3 {
                        x: 2.0,
                        y: 200.0,
                        z: 2.0,
                    },
                    Vec3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                    225.0,
                    -37.0,
                    0.0,
                ),
                16
            )),
        }
    }

    pub fn tick(&self, delta_time: f32) {
        let player_position;
        let render_distance;
        {
            let player = self.player.read();
            player_position = player.position;
            render_distance = player.render_distance;
        }
        let chunk_positions = Self::get_chunk_positions_square(player_position, render_distance as i32);
        let mut to_load = Vec::new();
        let mut to_unload = Vec::new();
        {
            let mut chunks = self.chunks.write();
            for chunk_position in chunk_positions.clone() {
                if !chunks.contains_key(&chunk_position) {
                    to_load.push(chunk_position);
                }
            }
            for chunk_position in chunks.keys() {
                if !chunk_positions.contains(chunk_position) {
                    to_unload.push(chunk_position.clone());
                }
            }
            for chunk_position in to_load.clone() {
                let chunk = self.generate_chunk(chunk_position);
                chunks.insert(chunk_position, chunk);
            }
        }
        for chunk_position in to_unload {
            self.sender
                .send(ChunkMessage::Unload(chunk_position))
                .unwrap();
            let mut chunks = self.chunks.write();
            chunks.remove(&chunk_position);
        }
        for chunk_position in to_load {
            let chunks = self.chunks.read();
            let chunk = chunks.get(&chunk_position).unwrap();
            self.sender
                .send(ChunkMessage::Load((
                    chunk_position,
                    self.get_chunk_meshes(chunk, &chunk_position),
                )))
                .unwrap();
        }
    }

    pub fn get_receiver(&self) -> Receiver<ChunkMessage> {
        self.receiver.clone()
    }

    fn world_to_chunk_pos(world_x: f32, world_z: f32) -> (i32, i32) {
        let chunk_x = (world_x / 16 as f32).floor() as i32;
        let chunk_z = (world_z / 16 as f32).floor() as i32;
        (chunk_x, chunk_z)
    }

    fn get_chunk_positions_square(player_pos: Vec3, render_distance_chunks: i32) -> Vec<IVec3> {
        let (chunk_x, chunk_z) = Self::world_to_chunk_pos(player_pos.x, player_pos.z);
        let mut chunks = Vec::new();

        let min_x = chunk_x - render_distance_chunks;
        let max_x = chunk_x + render_distance_chunks;
        let min_z = chunk_z - render_distance_chunks;
        let max_z = chunk_z + render_distance_chunks;

        for x in min_x..=max_x {
            for z in min_z..=max_z {
                chunks.push(IVec3 { x, y: 0, z });
            }
        }

        chunks
    }

    pub fn generate_chunk(&self, pos: IVec3) -> Chunk {
        let mut chunk = Chunk {
            block_ids: [0; 16 * 16 * 16],
        };

        for lx in 0..16 {
            for lz in 0..16 {
                let world_x = (pos.x * 16) + lx as i32;
                let world_z = (pos.z * 16) + lz as i32;
                let point = (self.fbm_perlin.get([world_x as f64, world_z as f64]) * 12.0) as u32;

                chunk.write_block_id(lx, point, lz, 1);
            }
        }
        chunk
    }

    pub fn get_block_id(&self, x: i32, y: i32, z: i32) -> Option<u32> {
        let chunk_x = x.rem_euclid(16) as usize;
        let chunk_y = y.rem_euclid(16) as usize;
        let chunk_z = z.rem_euclid(16) as usize;
        let chunks = self.chunks.read();
        chunks
            .get(&IVec3 {
                x: x.div_euclid(16),
                y: y.div_euclid(16),
                z: z.div_euclid(16),
            })
            .map(|chunk| chunk.get_block_id(chunk_x, chunk_y, chunk_z))
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
                Face::XPlus,
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
                Face::XMinus,
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
                Face::YPlus,
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
                Face::YMinus,
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
                Face::ZPlus,
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
                Face::ZMinus,
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
