pub mod renderer;
pub mod resources;

use std::{
    sync::{Arc, Mutex},
    thread,
};

use anyhow::{Ok, Result};
use winit::{
    event::{DeviceEvent, KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::PhysicalKey,
    window::{Window},
};

use crate::app::{
    engine::renderer::{Renderer, context::RenderingContext},
    game::{Game, world::World},
};

pub struct Engine {
    world: Arc<Mutex<World>>,
    renderer: Renderer,
    window: Arc<Window>,
    context: Arc<RenderingContext>,
}

impl Engine {
    pub fn new(event_loop: &ActiveEventLoop) -> Result<Self> {
        let world = Arc::new(Mutex::new(World::new()));
        let world_ = world.clone();
        thread::spawn(move || {
            let mut game = Game::new(world_);
            game.start_game();
        });

        let window = Arc::new(event_loop.create_window(Default::default())?);
        window.set_cursor_visible(false);
        window.set_cursor_grab(winit::window::CursorGrabMode::Locked)?;
        let context = Arc::new(RenderingContext::new(window.clone())?);
        let renderer = Renderer::new(context.clone(), window.clone(), world.clone())?;
        Ok(Self {
            world,
            renderer,
            window,
            context,
        })
    }

    pub fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.renderer
                    .process_camera_rotation(delta.0 as f32, delta.1 as f32);
            }
            DeviceEvent::Key(event) => match event.physical_key {
                PhysicalKey::Code(code) => {
                    self.renderer.process_camera_input(code, event.state);
                }
                _ => {}
            },
            _ => {}
        }
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
            WindowEvent::KeyboardInput {
                device_id,
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                is_synthetic,
            } => {
                //self.renderer.process_camera_input(code, key_state);
            }
            _ => {}
        }
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }
}
