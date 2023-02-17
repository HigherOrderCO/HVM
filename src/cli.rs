use crate::runtime;

pub fn parse_size(text: &str) -> Result<usize, String> {
  if text == "auto" {
    Ok(runtime::default_heap_size())
  } else {
    text.parse::<usize>().map_err(|x| format!("{}", x))
  }
}

pub fn parse_tids(text: &str) -> Result<usize, String> {
  if text == "auto" {
    Ok(runtime::default_heap_tids())
  } else {
    text.parse::<usize>().map_err(|x| format!("{}", x))
  }
}

pub fn parse_bool(text: &str) -> Result<bool, String> {
  text.parse::<bool>().map_err(|x| format!("{}", x))
}

pub fn load_code(file: &str) -> Result<String, String> {
  if file.is_empty() {
    Ok(String::new())
  } else {
    std::fs::read_to_string(file).map_err(|err| err.to_string())
  }
}