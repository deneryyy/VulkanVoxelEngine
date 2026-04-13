use glam::Vec3;

use crate::app::engine::renderer::vertex::Vertex;

pub struct Quad {
    pub vertices: [Vertex; 4],
}

#[derive(Eq, PartialEq, Hash, Clone, Copy)]
pub enum Face {
    ZPlus,
    ZMinus,
    YPlus,
    YMinus,
    XPlus,
    XMinus,
}

impl Quad {
    pub fn get_quad(face: Face, pos: Vec3) -> Quad {
        let mut quad = match face {
            Face::ZPlus => Quad {
                vertices: [
                    Vertex::new([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.0, 1.0),
                    Vertex::new([1.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0, 1.0),
                    Vertex::new([1.0, 1.0, 1.0], [0.0, 0.0, 1.0], 1.0, 0.0),
                    Vertex::new([0.0, 1.0, 1.0], [0.0, 0.0, 1.0], 0.0, 0.0),
                ],
            },
            Face::ZMinus => Quad {
                vertices: [
                    Vertex::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, 1.0),
                    Vertex::new([0.0, 1.0, 0.0], [0.0, 0.0, -1.0], 1.0, 0.0),
                    Vertex::new([1.0, 1.0, 0.0], [0.0, 0.0, -1.0], 0.0, 0.0),
                    Vertex::new([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0, 1.0),
                ],
            },
            Face::XPlus => Quad {
                vertices: [
                    Vertex::new([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 1.0),
                    Vertex::new([1.0, 1.0, 0.0], [1.0, 0.0, 0.0], 1.0, 0.0),
                    Vertex::new([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 0.0, 0.0),
                    Vertex::new([1.0, 0.0, 1.0], [1.0, 0.0, 0.0], 0.0, 1.0),
                ],
            },
            Face::XMinus => Quad {
                vertices: [
                    Vertex::new([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.0, 1.0),
                    Vertex::new([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], 1.0, 1.0),
                    Vertex::new([0.0, 1.0, 1.0], [-1.0, 0.0, 0.0], 1.0, 0.0),
                    Vertex::new([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], 0.0, 0.0),
                ],
            },
            Face::YPlus => Quad {
                vertices: [
                    Vertex::new([0.0, 1.0, 1.0], [0.0, 1.0, 0.0], 1.0, 0.0),
                    Vertex::new([1.0, 1.0, 1.0], [0.0, 1.0, 0.0], 0.0, 0.0),
                    Vertex::new([1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0.0, 1.0),
                    Vertex::new([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 1.0, 1.0),
                ],
            },
            Face::YMinus => Quad {
                vertices: [
                    Vertex::new([0.0, 0.0, 1.0], [0.0, -1.0, 0.0], 1.0, 0.0),
                    Vertex::new([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], 1.0, 1.0),
                    Vertex::new([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], 0.0, 1.0),
                    Vertex::new([1.0, 0.0, 1.0], [0.0, -1.0, 0.0], 0.0, 0.0),
                ],
            },
        };
        for vertex in &mut quad.vertices {
            vertex.pos[0] += pos.x;
            vertex.pos[1] += pos.y;
            vertex.pos[2] += pos.z;
        }
        return quad;
    }

    pub fn get_full_block_vertices() -> Vec<Vertex> {
        let mut vertices = Vec::new();
        vertices.extend_from_slice(
            &Quad::get_quad(
                Face::XPlus,
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .vertices,
        );
        vertices.extend_from_slice(
            &Quad::get_quad(
                Face::XMinus,
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .vertices,
        );
        vertices.extend_from_slice(
            &Quad::get_quad(
                Face::ZPlus,
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .vertices,
        );
        vertices.extend_from_slice(
            &Quad::get_quad(
                Face::ZMinus,
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .vertices,
        );
        vertices.extend_from_slice(
            &Quad::get_quad(
                Face::YPlus,
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .vertices,
        );
        vertices.extend_from_slice(
            &Quad::get_quad(
                Face::YMinus,
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .vertices,
        );
        vertices
    }
}
