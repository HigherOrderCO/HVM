use crate::runtime;

pub fn parse_size(text: &str) -> Result<usize, String> {
  if text == "auto" {
    return Ok(runtime::default_heap_size());
  } else {
    return text.parse::<usize>().map_err(|x| format!("{}", x));
  }
}

pub fn parse_tids(text: &str) -> Result<usize, String> {
  if text == "auto" {
    return Ok(runtime::default_heap_tids());
  } else {
    return text.parse::<usize>().map_err(|x| format!("{}", x));
  }
}

pub fn parse_bool(text: &str) -> Result<bool, String> {
  return text.parse::<bool>().map_err(|x| format!("{}", x));
}

pub fn load_code(file: &str) -> Result<String, String> {
  if file.is_empty() {
    return Ok(String::new());
  } else {
    return std::fs::read_to_string(file).map_err(|err| err.to_string());
  }
}
