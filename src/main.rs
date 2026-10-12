mod app;

use core::fmt;

use anyhow::Result;
use tracing_subscriber::EnvFilter;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::App;

fn main() -> Result<()> {
    let filter_layer = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter_layer)
        .init();
    let mut app = App::default();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app)?;
    Ok(())
}
