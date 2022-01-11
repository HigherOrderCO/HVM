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

// Types
// =====

#[derive(Clone, Copy, Debug)]
pub struct State<'a> {
  pub code: &'a str,
  pub index: usize,
}

pub type Answer<'a, A> = Result<(State<'a>, A), String>;
pub type Parser<'a, A> = Box<dyn Fn(State<'a>) -> Answer<'a, A>>;

// Utils
// =====

pub fn equal_at(text: &str, test: &str, i: usize) -> bool {
  return &text[i .. std::cmp::min(text.len(), i + test.len())] == test;
}

pub fn flatten(texts: &[&str]) -> String {
  let mut result: String = String::new();
  for text in texts.iter() {
    for chr in text.chars() {
      result.push(chr);
    }
  }
  return result;
}

pub fn lines(text: &str) -> Vec<String> {
  let mut result: Vec<String> = Vec::new();
  for line in text.lines() {
    result.push(line.to_string());
  }
  return result;
}

pub fn find(text: &str, target: &str) -> usize {
  return text.find(target).unwrap();
}

pub fn read<'a, A>(parser: Parser<'a, A>, code: &'a str) -> A {
  match parser(State { code, index: 0 }) {
    Ok((state, value)) => value,
    Err(msg) => {
      println!("{}", msg);
      panic!("No parse.");
    },
  }
}

// Elims
// =====

pub fn head<'a>(state: State<'a>) -> Option<char> {
  return state.code[state.index..].chars().next();
}

pub fn head_default<'a>(state: State<'a>) -> char {
  if let Some(got) = head(state) {
    return got;
  } else {
    return '\0';
  }
}

pub fn tail<'a>(state: State<'a>) -> State {
  let fst = head(state);
  let add = match head(state) {
    Some(c) => c.len_utf8(),
    None => 0,
  };
  return State {
    code: state.code,
    index: state.index + add,
  };
}

pub fn get_char<'a>(state: State<'a>) -> Answer<'a, char> {
  let (state, skipped) = skip(state)?;
  if let Some(got) = head(state) {
    let state = State {
      code: state.code,
      index: state.index + got.len_utf8(),
    };
    return Ok((state, got));
  } else {
    return Ok((state, '\0'));
  }
}

pub fn get_char_parser<'a>() -> Parser<'a, char> {
  return Box::new(|x| get_char(x));
}

// Skippers
// ========

pub fn skip_comment<'a>(state: State<'a>) -> Answer<'a, bool> {
  let mut state = state;
  if state.index + 1 < state.code.len() && equal_at(&state.code, "//", state.index) {
    state.index += 2;
    while state.index < state.code.len() && equal_at(&state.code, "\n", state.index) {
      state.index += 1;
    }
    return Ok((state, true));
  } else {
    return Ok((state, false));
  };
}

pub fn skip_comment_parser<'a>() -> Parser<'a, bool> {
  return Box::new(|x| skip_comment(x));
}

pub fn skip_spaces<'a>(state: State<'a>) -> Answer<'a, bool> {
  pub fn is_space(chr: char) -> bool {
    return chr == ' ' || chr == '\n' || chr == '\t';
  }
  let mut state = state;
  if state.index < state.code.len() && is_space(head_default(state)) {
    state.index += 1;
    while state.index < state.code.len() && is_space(head_default(state)) {
      state.index += 1;
    }
    return Ok((state, true));
  } else {
    return Ok((state, false));
  }
}

pub fn skip_spaces_parser<'a>() -> Parser<'a, bool> {
  return Box::new(|x| skip_spaces(x));
}

pub fn skip<'a>(state: State<'a>) -> Answer<'a, bool> {
  let (state, comment) = skip_comment(state)?;
  let (state, spaces) = skip_spaces(state)?;
  if comment || spaces {
    let (state, skipped) = skip(state)?;
    return Ok((state, true));
  } else {
    return Ok((state, false));
  }
}

pub fn skip_parser<'a>() -> Parser<'a, bool> {
  return Box::new(|x| skip(x));
}

// Strings
// =======

// Attempts to match a string right after the cursor.
// Returns true if successful. Consumes string.
pub fn text_here<'a>(pat: &str, state: State<'a>) -> Answer<'a, bool> {
  if equal_at(&state.code, pat, state.index) {
    let state = State {
      code: state.code,
      index: state.index + pat.len(),
    };
    return Ok((state, true));
  } else {
    return Ok((state, false));
  }
}

pub fn text_here_parser<'a>(pat: &'static str) -> Parser<'a, bool> {
  return Box::new(move |x| text_here(pat, x));
}

// Like 'text_here', but skipping spaces and comments before.
pub fn text<'a>(pat: &str, state: State<'a>) -> Answer<'a, bool> {
  let (state, skipped) = skip(state)?;
  let (state, matched) = text_here(pat, state)?;
  return Ok((state, matched));
}

pub fn text_parser<'a>(pat: &'static str) -> Parser<'a, bool> {
  return Box::new(move |x| text(pat, x));
}

// Like 'text', but aborts if there is no match.
pub fn consume<'a>(pat: &str, state: State<'a>) -> Answer<'a, ()> {
  let (state, matched) = text(pat, state)?;
  if matched {
    return Ok((state, ()));
  } else {
    return expected(pat, pat.len(), state);
  }
}

pub fn consume_parser<'a>(pat: &'static str) -> Parser<'a, ()> {
  return Box::new(move |x| consume(pat, x));
}

// Returns true if we are at the end of the file, skipping spaces and comments.
pub fn done<'a>(state: State<'a>) -> Answer<'a, bool> {
  let (state, skipped) = skip(state)?;
  return Ok((state, state.index == state.code.len()));
}

pub fn done_parser<'a>() -> Parser<'a, bool> {
  return Box::new(move |x| done(x));
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
  let (_    , matched) = head(state)?;
  if matched {
    let (state, got) = body(state)?;
    return Ok((state, Some(got)));
  } else {
    return Ok((state, None));
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
  return expected(name, 1, state);
}

// Combinators
// ===========

// Evaluates a parser and returns its result, but reverts its effect.
pub fn dry<'a, A: 'a>(parser: Parser<'a, A>, state: State<'a>) -> Answer<'a, A> {
  let (new_state, result) = parser(state)?;
  return Ok((state, result));
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
  return Ok((state, result));
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
    let (new_state, skip) = parse_sep(state)?;
    if done {
      break;
    } else {
      let (new_state, elem) = parse_elem(new_state)?;
      state = new_state;
      elems.push(elem);
    }
  }
  return Ok((state, make(elems)));
}

// Name
// ====

// Is this character a valid name letter?
fn is_letter(chr: char) -> bool {
  return (chr >= 'A' && chr <= 'Z'
    || chr >= 'a' && chr <= 'z'
    || chr >= '0' && chr <= '9'
    || chr == '_'
    || chr == '.');
}

// Parses a name right after the parsing cursor.
pub fn name_here<'a>(state: State<'a>) -> Answer<'a, String> {
  let mut name: String = String::new();
  let mut state = state;
  loop {
    if let Some(got) = head(state) {
      if is_letter(got) {
        name.push(got);
        state = tail(state);
      } else {
        break;
      }
    } else {
      break;
    }
  }
  return Ok((state, name));
}

// Parses a name after skipping.
pub fn name<'a>(state: State<'a>) -> Answer<'a, String> {
  let (state, skipped) = skip(state)?;
  return name_here(state);
}

// Parses a non-empty name after skipping.
pub fn name1<'a>(state: State<'a>) -> Answer<'a, String> {
  let (state, name1) = name(state)?;
  if name1.len() > 0 {
    return Ok((state, name1));
  } else {
    return expected("name", 1, state);
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
      rest = flatten(&[&open_color, &line, &close_color, "\n"]);
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
  return text;
}

// Tests
// =====

pub enum Testree {
  Node {
    lft: Box<Testree>,
    rgt: Box<Testree>,
  },
  Leaf {
    val: String,
  },
}

pub fn testree_show(tt: &Testree) -> String {
  match tt {
    Testree::Node { lft, rgt } => format!("({} {})", testree_show(lft), testree_show(rgt)),
    Testree::Leaf { val } => format!("{}", val),
  }
}

pub fn node_parser<'a>() -> Parser<'a, Option<Box<Testree>>> {
  return Box::new(|state| {
    return guard(
      text_parser("("),
      Box::new(|state| {
        let (state, skp) = consume("(", state)?;
        let (state, lft) = testree_parser()(state)?;
        let (state, rgt) = testree_parser()(state)?;
        let (state, skp) = consume(")", state)?;
        return Ok((state, Box::new(Testree::Node { lft, rgt })));
      }),
      state,
    );
  });
}

pub fn leaf_parser<'a>() -> Parser<'a, Option<Box<Testree>>> {
  return Box::new(|state| {
    return guard(
      text_parser(""),
      Box::new(|state| {
        let (state, val) = name(state)?;
        return Ok((state, Box::new(Testree::Leaf { val })));
      }),
      state,
    );
  });
}

pub fn testree_parser<'a>() -> Parser<'a, Box<Testree>> {
  return Box::new(|state| {
    let (state, tree) = grammar("Testree", &[node_parser(), leaf_parser()], state)?;
    return Ok((state, tree));
  });
}
