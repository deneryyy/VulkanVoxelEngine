use glam::Vec3;

use crate::app::engine::renderer::vertex::Vertex;

pub struct Quad {
    pub vertices: [Vertex; 4],
}

#[derive(Eq, PartialEq, Hash, Clone, Copy)]
pub enum Face {
    Z_PLUS,
    Z_MINUS,
    Y_PLUS,
    Y_MINUS,
    X_PLUS,
    X_MINUS,
}

impl Quad {
    pub fn get_quad(face: Face, pos: Vec3) -> Quad {
        let mut quad = match face {
            Face::Z_PLUS => Quad {
                vertices: [
                    Vertex::new([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.0, 1.0),
                    Vertex::new([1.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0, 1.0),
                    Vertex::new([1.0, 1.0, 1.0], [0.0, 0.0, 1.0], 1.0, 0.0),
                    Vertex::new([0.0, 1.0, 1.0], [0.0, 0.0, 1.0], 0.0, 0.0),
                ],
            },
            Face::Z_MINUS => Quad {
                vertices: [
                    Vertex::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, 1.0),
                    Vertex::new([0.0, 1.0, 0.0], [0.0, 0.0, -1.0], 1.0, 0.0),
                    Vertex::new([1.0, 1.0, 0.0], [0.0, 0.0, -1.0], 0.0, 0.0),
                    Vertex::new([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0, 1.0),
                ],
            },
            Face::X_PLUS => Quad {
                vertices: [
                    Vertex::new([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 1.0),
                    Vertex::new([1.0, 1.0, 0.0], [1.0, 0.0, 0.0], 1.0, 0.0),
                    Vertex::new([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 0.0, 0.0),
                    Vertex::new([1.0, 0.0, 1.0], [1.0, 0.0, 0.0], 0.0, 1.0),
                ],
            },
            Face::X_MINUS => Quad {
                vertices: [
                    Vertex::new([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.0, 1.0),
                    Vertex::new([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], 1.0, 1.0),
                    Vertex::new([0.0, 1.0, 1.0], [-1.0, 0.0, 0.0], 1.0, 0.0),
                    Vertex::new([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], 0.0, 0.0),
                ],
            },
            Face::Y_PLUS => Quad {
                vertices: [
                    Vertex::new([0.0, 1.0, 1.0], [0.0, 1.0, 0.0], 1.0, 0.0),
                    Vertex::new([1.0, 1.0, 1.0], [0.0, 1.0, 0.0], 0.0, 0.0),
                    Vertex::new([1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0.0, 1.0),
                    Vertex::new([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 1.0, 1.0),
                ],
            },
            Face::Y_MINUS => Quad {
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
                Face::X_PLUS,
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
                Face::X_MINUS,
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
                Face::Z_PLUS,
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
                Face::Z_MINUS,
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
                Face::Y_PLUS,
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
                Face::Y_MINUS,
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
