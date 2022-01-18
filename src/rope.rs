use ropey::Rope;
use std::{fmt, fmt::Write};

#[derive(Debug, Clone, Default)]
pub struct RopeBuilder {
  rope_builder: ropey::RopeBuilder,
}

impl RopeBuilder {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn append(&mut self, chunk: &str) {
    self.rope_builder.append(chunk);
  }

  pub fn finish(self) -> Rope {
    self.rope_builder.finish()
  }
}

impl Write for RopeBuilder {
  fn write_str(&mut self, s: &str) -> fmt::Result {
    self.append(s);
    Ok(())
  }
}
