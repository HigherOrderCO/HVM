pub type Text = Vec<char>;

//impl std::fmt::Display for Text {
  //fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    //let mut result = String::new();
    //for chr in self.iter() {
      //result.push(*chr);
    //}
    //write!(f, "{}", result)
  //}
//}

pub fn equal_at(text: &Text, test: &Text, i: usize) -> bool {
  for j in 0..test.len() {
    if test[i as usize + j] != test[j as usize] {
      return false;
    }
  }
  return true;
}

pub fn text_to_utf8(text: &[char]) -> String {
  return text.iter().collect();
}

pub fn utf8_to_text(string: &str) -> Text {
  return string.chars().collect();
}

pub fn flatten(texts: &[&[char]]) -> Vec<char> {
  let mut result: Text = Vec::new();
  for text in texts.iter() { 
    for chr in text.iter() {
      result.push(*chr);
    }
  }
  return result;
}

pub fn lines(text: &[char]) -> Vec<Vec<char>> {
  let mut result : Vec<Text> = Vec::new();
  let mut line : Text = Vec::new();
  for chr in text {
    if chr == &'\n' {
      result.push(line);
      line = Vec::new();
    } else {
      line.push(*chr);
    }
  }
  if (line.len() > 0) {
    result.push(line);
  }
  return result;
}

pub fn find(text: &[char], target: char) -> usize {
  for i in 0..text.len() {
    if target == text[i] {
      return i;
    }
  }
  return text.len();
}

// TODO: This was ported from JavaScript, where string slicing is fast and idiomatic. In Rust,
// strings are often represented as UTF-8, so `str.slice(i,j)` doesn't work the same as JS. As
// such, this core is in poor shape and must be improved.
pub fn highlight(from_index: usize, to_index: usize, code: &Text) -> Text {
  let open = '«';
  let close = '»';
  let open_color = utf8_to_text("\x1b[4m\x1b[31m");
  let close_color = utf8_to_text("\x1b[0m");
  let mut from_line = 0;
  let mut to_line = 0;
  for (i, c) in code.iter().enumerate() {
    if *c == '\n' {
      if i < from_index { from_line += 1; }
      if i < to_index { to_line += 1; }
    }
  }
  let code : Text = flatten(&[
    &code[0..from_index],
    &[open],
    &code[from_index .. to_index],
    &[close],
    &code[to_index .. code.len()],
  ]);
  let lines : Vec<Text> = lines(&code);
  let block_from_line = std::cmp::max(from_line as i64 - 3, 0) as usize;
  let block_to_line = std::cmp::min(to_line + 3, lines.len());
  let mut text = Vec::new();
  for (i, line) in lines[block_from_line .. block_to_line].iter().enumerate() {
    let numb = block_from_line + i;
    let rest;
    if numb == from_line && numb == to_line {
      rest = flatten(&[
        &line[0 .. find(line, open)],
        &open_color,
        &line[find(line, open) .. find(line, close) + 1],
        &close_color,
        &line[find(line, close) + 1 .. line.len()],
        &['\n'],
      ]);
    } else if numb == from_line {
      rest = flatten(&[
        &line[0 .. find(line, open)],
        &open_color,
        &line[find(line, open) .. line.len()],
        &['\n'],
      ]);
    } else if numb > from_line && numb < to_line {
      rest = flatten(&[
        &open_color,
        &line,
        &close_color,
        &['\n'],
      ]);
    } else if numb == to_line {
      rest = flatten(&[
        &line[0 .. find(line, open)],
        &open_color,
        &line[find(line, open) .. find(line, close) + 1],
        &close_color,
        &['\n'],
      ]);
    } else {
      rest = flatten(&[
        &line,
        &['\n'],
      ]);
    }
    let line = utf8_to_text(&format!("    {} | {}", numb, &text_to_utf8(&rest)));
    text.append(&mut line.clone());
  }
  return text;
}
