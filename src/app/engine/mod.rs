pub mod renderer;

use std::sync::Arc;

use anyhow::{Ok, Result};
use winit::{
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::app::engine::renderer::{Renderer, context::RenderingContext};

pub struct Engine {
    renderer: Renderer,
    window: Arc<Window>,
    context: Arc<RenderingContext>,
}

impl Engine {
    pub fn new(event_loop: &ActiveEventLoop) -> Result<Self> {
        let window = Arc::new(event_loop.create_window(Default::default())?);
        let context = Arc::new(RenderingContext::new(window.clone())?);
        let renderer = Renderer::new(context.clone(), window.clone())?;
        Ok(Self {
            renderer,
            window,
            context,
        })
    }

    pub fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.renderer.resize().unwrap();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                self.renderer.resize().unwrap();
            }
            WindowEvent::RedrawRequested => {
                self.renderer.render().unwrap();
            }
            _ => {}
        }
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }
}
