use std::sync::LazyLock;


#[derive(Clone, Copy, PartialEq)]
pub enum TerrainRenderMode {
    Vertex,
    ComputeRayTracing,
}

pub struct GameSettings {
    pub terrain_render_mode: TerrainRenderMode
}

pub static GAME_SETTINGS: LazyLock<GameSettings> = LazyLock::new(|| {
    GameSettings {
        terrain_render_mode: TerrainRenderMode::ComputeRayTracing
    }
});