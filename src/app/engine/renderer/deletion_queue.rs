pub struct DeletionQueue {
    queue: Vec<Box<dyn Fn()>>,
}

impl DeletionQueue {
    pub fn new() -> Self {
        return Self { queue: Vec::new() };
    }

    pub fn push_function(&mut self, function: Box<dyn Fn()>) {
        self.queue.push(function);
    }

    pub fn flush(&mut self) {
        self.queue.drain(..).for_each(|el| el());
    }
}
