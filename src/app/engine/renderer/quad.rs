use glam::Vec3;

use crate::app::engine::renderer::vertex::Vertex;

pub struct Quad {
    pub vertices: [Vertex; 4],
}

pub enum Direction {
    Z_PLUS,
    Z_MINUS,
    Y_PLUS,
    Y_MINUS,
    X_PLUS,
    X_MINUS,
}

impl Quad {
    pub fn get_quad(direction: Direction, pos: Vec3) -> Quad {
        let mut quad = match direction {
            Direction::Z_PLUS => Quad {
                vertices: [
                    Vertex::new([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.0, 1.0),
                    Vertex::new([1.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0, 1.0),
                    Vertex::new([1.0, 1.0, 1.0], [0.0, 0.0, 1.0], 1.0, 0.0),
                    Vertex::new([0.0, 1.0, 1.0], [0.0, 0.0, 1.0], 0.0, 0.0),
                ],
            },
            Direction::Z_MINUS => Quad {
                vertices: [
                    Vertex::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, 1.0),
                    Vertex::new([0.0, 1.0, 0.0], [0.0, 0.0, -1.0], 1.0, 0.0),
                    Vertex::new([1.0, 1.0, 0.0], [0.0, 0.0, -1.0], 0.0, 0.0),
                    Vertex::new([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], 0.0, 1.0),
                ],
            },
            Direction::X_PLUS => Quad {
                vertices: [
                    Vertex::new([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0, 1.0),
                    Vertex::new([1.0, 1.0, 0.0], [1.0, 0.0, 0.0], 1.0, 0.0),
                    Vertex::new([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 0.0, 0.0),
                    Vertex::new([1.0, 0.0, 1.0], [1.0, 0.0, 0.0], 0.0, 1.0),
                ],
            },
            Direction::X_MINUS => Quad {
                vertices: [
                    Vertex::new([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 0.0, 1.0),
                    Vertex::new([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], 1.0, 1.0),
                    Vertex::new([0.0, 1.0, 1.0], [-1.0, 0.0, 0.0], 1.0, 0.0),
                    Vertex::new([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], 0.0, 0.0),
                ],
            },
            Direction::Y_PLUS => Quad {
                vertices: [
                    Vertex::new([0.0, 1.0, 1.0], [0.0, 1.0, 0.0], 1.0, 0.0),
                    Vertex::new([1.0, 1.0, 1.0], [0.0, 1.0, 0.0], 0.0, 0.0),
                    Vertex::new([1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0.0, 1.0),
                    Vertex::new([0.0, 1.0, 0.0], [0.0, 1.0, 0.0], 1.0, 1.0),
                ],
            },
            Direction::Y_MINUS => Quad {
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
}
