// This parse library is more high-level and functional than existing alternatives.
// A Parser is defied as (with details omitted):
//
//   Answer<A> = Result<(State, A), String>
//   Parser<A> = Fn(State) -> Answer<A>>
//
// Similarly to https://github.com/AndrasKovacs/flatparse, there are 2 ways to fail.
//
// 1. Recoverable. Use Parser<Option<A>>, and return:
//    - Ok((new_state, Some(result))) if it succeeds
//    - Ok((old_state, None))         if it fails
//
//    This backtracks, and should be used to implement alternatives. For example, if you're
//    parsing an AST, "Animal", with 2 constructors, dog and cat, then you should implement:
//
//    parse_dog    : Parser<Option<Animal>>
//    parse_cat    : Parser<Option<Animal>>
//    parse_animal : Parser<Animal>
//
// 2. Irrecoverable. Return:
//    - Err(error_message)
//
//    This will abort the entire parser, like a "throw", and return the error message. Use this
//    when you know that only one parsing branch can reach this location, yet the source is wrong.
//
// Check the Testree example at the bottom of this file.

use std::cmp;

// Types
// =====

#[derive(Clone, Copy, Debug)]
pub struct State<'a> {
  pub code: &'a str,
  pub index: usize,
}

impl<'a> State<'a> {
  fn rest(&self) -> &'a str {
    &self.code[self.index..]
  }
}

pub type Answer<'a, A> = Result<(State<'a>, A), String>;
pub type Parser<'a, A> = Box<dyn Fn(State<'a>) -> Answer<'a, A>>;

// Utils
// =====

pub fn equal_at(text: &str, test: &str, i: usize) -> bool {
  return &text.as_bytes()[i..std::cmp::min(text.len(), i + test.len())] == test.as_bytes();
}

pub fn flatten(texts: &[&str]) -> String {
  texts.concat()
}

pub fn lines(text: &str) -> Vec<String> {
  text.lines().map(String::from).collect()
}

pub fn find(text: &str, target: &str) -> usize {
  text.find(target).unwrap()
}

pub fn read<'a, A>(parser: Parser<'a, A>, code: &'a str) -> A {
  match parser(State { code, index: 0 }) {
    Ok((state, value)) => value,
    Err(msg) => {
      println!("{}", msg);
      panic!("No parse.");
    }
  }
}

// Elims
// =====

pub fn head(state: State) -> Option<char> {
  return state.code[state.index..].chars().next();
}

pub fn head_default(state: State) -> char {
  if let Some(got) = head(state) {
    got
  } else {
    '\0'
  }
}

pub fn tail(state: State) -> State {
  let fst = head(state);
  let add = match head(state) {
    Some(c) => c.len_utf8(),
    None => 0,
  };
  State { code: state.code, index: state.index + add }
}

pub fn get_char(state: State) -> Answer<char> {
  let (state, skipped) = skip(state)?;
  if let Some(got) = head(state) {
    let state = State { code: state.code, index: state.index + got.len_utf8() };
    Ok((state, got))
  } else {
    Ok((state, '\0'))
  }
}

pub fn get_char_parser<'a>() -> Parser<'a, char> {
  Box::new(get_char)
}

// Skippers
// ========

pub fn skip_comment(state: State) -> Answer<bool> {
  let mut state = state;
  if state.index + 1 < state.code.len() && equal_at(state.code, "//", state.index) {
    state.index += 2;
    while state.index < state.code.len() && !equal_at(state.code, "\n", state.index) {
      state.index += 1;
    }
    Ok((state, true))
  } else {
    Ok((state, false))
  }
}

pub fn skip_comment_parser<'a>() -> Parser<'a, bool> {
  Box::new(skip_comment)
}

pub fn skip_spaces(state: State) -> Answer<bool> {
  pub fn is_space(chr: char) -> bool {
    chr == ' ' || chr == '\n' || chr == '\t' || chr == '\r'
  }
  let mut state = state;
  if state.index < state.code.len() && is_space(head_default(state)) {
    state.index += 1;
    while state.index < state.code.len() && is_space(head_default(state)) {
      state.index += 1;
    }
    Ok((state, true))
  } else {
    Ok((state, false))
  }
}

pub fn skip_spaces_parser<'a>() -> Parser<'a, bool> {
  Box::new(skip_spaces)
}

pub fn skip(state: State) -> Answer<bool> {
  let (state, comment) = skip_comment(state)?;
  let (state, spaces) = skip_spaces(state)?;
  if comment || spaces {
    let (state, skipped) = skip(state)?;
    Ok((state, true))
  } else {
    Ok((state, false))
  }
}

pub fn skip_parser<'a>() -> Parser<'a, bool> {
  Box::new(skip)
}

// Strings
// =======

// Attempts to match a string right after the cursor.
// Returns true if successful. Consumes string.
pub fn text_here<'a>(pat: &str, state: State<'a>) -> Answer<'a, bool> {
  if equal_at(state.code, pat, state.index) {
    let state = State { code: state.code, index: state.index + pat.len() };
    Ok((state, true))
  } else {
    Ok((state, false))
  }
}

pub fn text_here_parser<'a>(pat: &'static str) -> Parser<'a, bool> {
  Box::new(move |x| text_here(pat, x))
}

// Like 'text_here', but skipping spaces and comments before.
pub fn text<'a>(pat: &str, state: State<'a>) -> Answer<'a, bool> {
  let (state, skipped) = skip(state)?;
  let (state, matched) = text_here(pat, state)?;
  Ok((state, matched))
}

pub fn text_parser<'a>(pat: &'static str) -> Parser<'a, bool> {
  Box::new(move |x| text(pat, x))
}

// Like 'text', but aborts if there is no match.
pub fn consume<'a>(pat: &str, state: State<'a>) -> Answer<'a, ()> {
  let (state, matched) = text(pat, state)?;
  if matched {
    Ok((state, ()))
  } else {
    expected(pat, pat.len(), state)
  }
}

pub fn consume_parser<'a>(pat: &'static str) -> Parser<'a, ()> {
  Box::new(move |x| consume(pat, x))
}

// Returns true if we are at the end of the file, skipping spaces and comments.
pub fn done(state: State) -> Answer<bool> {
  let (state, skipped) = skip(state)?;
  Ok((state, state.index == state.code.len()))
}

pub fn done_parser<'a>() -> Parser<'a, bool> {
  Box::new(done)
}

// Blocks
// ======

// Checks if a dry-run of the first parser returns true.
// If so, applies the second parser, returning Some.
// If no, return None.
pub fn guard<'a, A: 'a>(
  head: Parser<'a, bool>,
  body: Parser<'a, A>,
  state: State<'a>,
) -> Answer<'a, Option<A>> {
  let (state, skipped) = skip(state)?;
  let (_, matched) = head(state)?;
  if matched {
    let (state, got) = body(state)?;
    Ok((state, Some(got)))
  } else {
    Ok((state, None))
  }
}

// Applies optional parsers in sequence.
// Returns the first that succeeds.
// If none succeeds, aborts.
pub fn grammar<'a, A: 'a>(
  name: &'static str,
  choices: &[Parser<'a, Option<A>>],
  state: State<'a>,
) -> Answer<'a, A> {
  for choice in choices {
    let (state, result) = choice(state)?;
    if let Some(value) = result {
      return Ok((state, value));
    }
  }
  expected(name, 1, state)
}

// Combinators
// ===========

// Evaluates a parser and returns its result, but reverts its effect.
pub fn dry<'a, A: 'a>(parser: Parser<'a, A>, state: State<'a>) -> Answer<'a, A> {
  let (new_state, result) = parser(state)?;
  Ok((state, result))
}

// Evaluates a parser until a condition is met. Returns an array of results.
pub fn until<'a, A: 'a>(
  delim: Parser<'a, bool>,
  parser: Parser<'a, A>,
  state: State<'a>,
) -> Answer<'a, Vec<A>> {
  let mut state = state;
  let mut result = Vec::new();
  loop {
    let (new_state, delimited) = delim(state)?;
    if delimited {
      state = new_state;
      break;
    } else {
      let (new_state, a) = parser(new_state)?;
      state = new_state;
      result.push(a);
    }
  }
  Ok((state, result))
}

// Evaluates a list-like parser, with an opener, separator, and closer.
pub fn list<'a, A: 'a, B: 'a>(
  parse_open: Parser<'a, bool>,
  parse_sep: Parser<'a, bool>,
  parse_close: Parser<'a, bool>,
  parse_elem: Parser<'a, A>,
  make: Box<dyn Fn(Vec<A>) -> B>,
  state: State<'a>,
) -> Answer<'a, B> {
  let (state, skp) = parse_open(state)?;
  let mut state = state;
  let mut elems = Vec::new();
  loop {
    let (new_state, done) = parse_close(state)?;
    let (new_state, skip) = parse_sep(new_state)?;
    if done {
      state = new_state;
      break;
    } else {
      let (new_state, elem) = parse_elem(new_state)?;
      state = new_state;
      elems.push(elem);
    }
  }
  Ok((state, make(elems)))
}

// Name
// ====

// Is this character a valid name letter?
fn is_letter(chr: char) -> bool {
  ('A'..='Z').contains(&chr)
    || ('a'..='z').contains(&chr)
    || ('0'..='9').contains(&chr)
    || chr == '_'
    || chr == '.'
}

// Parses a name right after the parsing cursor.
pub fn name_here(state: State) -> Answer<String> {
  let mut name: String = String::new();
  let mut state = state;
  while let Some(got) = head(state) {
    if is_letter(got) {
      name.push(got);
      state = tail(state);
    } else {
      break;
    }
  }
  Ok((state, name))
}

// Parses a name after skipping.
pub fn name(state: State) -> Answer<String> {
  let (state, skipped) = skip(state)?;
  name_here(state)
}

// Parses a non-empty name after skipping.
pub fn name1(state: State) -> Answer<String> {
  let (state, name1) = name(state)?;
  if !name1.is_empty() {
    Ok((state, name1))
  } else {
    expected("name", 1, state)
  }
}

// Errors
// ======

pub fn expected<'a, A>(name: &str, size: usize, state: State<'a>) -> Answer<'a, A> {
  return Err(format!(
    "Expected {}:\n{}",
    name,
    &highlight(state.index, state.index + size, state.code)
  ));
}

pub fn highlight(from_index: usize, to_index: usize, code: &str) -> String {
  //let open = "<<<<####";
  //let close = "####>>>>";
  let open = "««««";
  let close = "»»»»";
  let open_color = "\x1b[4m\x1b[31m";
  let close_color = "\x1b[0m";
  let mut from_line = 0;
  let mut to_line = 0;
  for (i, c) in code.chars().enumerate() {
    if c == '\n' {
      if i < from_index {
        from_line += 1;
      }
      if i < to_index {
        to_line += 1;
      }
    }
  }
  let code: String = flatten(&[
    &code[0..from_index],
    open,
    &code[from_index..to_index],
    close,
    &code[to_index..code.len()],
  ]);
  let lines: Vec<String> = lines(&code);
  let block_from_line = std::cmp::max(from_line as i64 - 3, 0) as usize;
  let block_to_line = std::cmp::min(to_line + 3, lines.len());
  let mut text = String::new();
  for (i, line) in lines[block_from_line..block_to_line].iter().enumerate() {
    let numb = block_from_line + i;
    let rest;
    if numb == from_line && numb == to_line {
      rest = flatten(&[
        &line[0..find(line, open)],
        open_color,
        &line[find(line, open) + open.len()..find(line, close)],
        close_color,
        &line[find(line, close) + close.len()..line.len()],
        "\n",
      ]);
    } else if numb == from_line {
      rest = flatten(&[
        &line[0..find(line, open)],
        open_color,
        &line[find(line, open)..line.len()],
        "\n",
      ]);
    } else if numb > from_line && numb < to_line {
      rest = flatten(&[open_color, line, close_color, "\n"]);
    } else if numb == to_line {
      rest = flatten(&[
        &line[0..find(line, open)],
        open_color,
        &line[find(line, open)..find(line, close) + close.len()],
        close_color,
        "\n",
      ]);
    } else {
      rest = flatten(&[line, "\n"]);
    }
    let line = format!("    {} | {}", numb, rest);
    text.push_str(&line);
  }
  text
}

// Tests
// =====

pub enum Testree {
  Node { lft: Box<Testree>, rgt: Box<Testree> },
  Leaf { val: String },
}

pub fn testree_show(tt: &Testree) -> String {
  match tt {
    Testree::Node { lft, rgt } => format!("({} {})", testree_show(lft), testree_show(rgt)),
    Testree::Leaf { val } => val.to_string(),
  }
}

pub fn node_parser<'a>() -> Parser<'a, Option<Box<Testree>>> {
  Box::new(|state| {
    guard(
      text_parser("("),
      Box::new(|state| {
        let (state, skp) = consume("(", state)?;
        let (state, lft) = testree_parser()(state)?;
        let (state, rgt) = testree_parser()(state)?;
        let (state, skp) = consume(")", state)?;
        Ok((state, Box::new(Testree::Node { lft, rgt })))
      }),
      state,
    )
  })
}

pub fn leaf_parser<'a>() -> Parser<'a, Option<Box<Testree>>> {
  Box::new(|state| {
    guard(
      text_parser(""),
      Box::new(|state| {
        let (state, val) = name(state)?;
        Ok((state, Box::new(Testree::Leaf { val })))
      }),
      state,
    )
  })
}

pub fn testree_parser<'a>() -> Parser<'a, Box<Testree>> {
  Box::new(|state| {
    let (state, tree) = grammar("Testree", &[node_parser(), leaf_parser()], state)?;
    Ok((state, tree))
  })
}
