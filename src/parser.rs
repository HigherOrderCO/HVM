// This parse library is more high-level and functional than existing alternatives.
// A Parser is defined as (with details omitted):
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

#![allow(dead_code)]
use itertools::Itertools;

// Types
// =====

#[derive(Clone, Copy, Debug)]
pub struct State<'a> {
  pub code: &'a str,
  pub index: usize,
}

impl<'a> State<'a> {
  fn rest(&self) -> Option<&'a str> {
    self.code.get(self.index..)
  }
}

pub type Answer<'a, A> = Result<(State<'a>, A), String>;
pub type Parser<'a, A> = Box<dyn Fn(State<'a>) -> Answer<'a, A>>;

// Utils
// =====

pub fn find(text: &str, target: &str) -> usize {
  text.find(target).unwrap_or_else(|| panic!("`{}` not in `{}`.", target, text))
}

pub fn read<'a, A>(parser: Parser<'a, A>, code: &'a str) -> A {
  match parser(State { code, index: 0 }) {
    Ok((_, value)) => value,
    Err(msg) => {
      println!("{}", msg);
      panic!("No parse.");
    }
  }
}

// Elims
// =====

/// Maybe gets the current character.
pub fn head(state: State) -> Option<char> {
  state.rest()?.chars().next()
}

/// Skips the current character.
pub fn tail(state: State) -> State {
  let add = match head(state) {
    Some(c) => c.len_utf8(),
    None => 0,
  };
  // NOTE: Could just mutate `state.index` here?
  State { code: state.code, index: state.index + add }
}

/// Skips comments and whitespace, then returns the next `char`, or the null
/// character if this doesn't exist.
pub fn get_char(state: State) -> Answer<char> {
  let (state, _) = skip(state)?;
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

pub fn skip_comment(mut state: State) -> Answer<bool> {
  const COMMENT: &str = "//";
  if let Some(rest) = state.rest() {
    if let Some(line) = rest.lines().next() {
      if line.starts_with(COMMENT) {
        state.index += line.len();
        return Ok((state, true));
      }
    }
  }
  Ok((state, false))
}

pub fn skip_comment_parser<'a>() -> Parser<'a, bool> {
  Box::new(skip_comment)
}

pub fn skip_spaces(mut state: State) -> Answer<bool> {
  if let Some(rest) = state.rest() {
    let add: usize = rest.chars().take_while(|a| a.is_whitespace()).map(|a| a.len_utf8()).sum();
    state.index += add;
    if add > 0 {
      return Ok((state, true));
    }
  }
  Ok((state, false))
}

pub fn skip_spaces_parser<'a>() -> Parser<'a, bool> {
  Box::new(skip_spaces)
}

/// Skips comments and whitespace.
pub fn skip(mut state: State) -> Answer<bool> {
  let (new_state, mut comment) = skip_comment(state)?;
  state = new_state;
  let (new_state, mut spaces) = skip_spaces(state)?;
  state = new_state;
  if comment || spaces {
    loop {
      let (new_state, new_comment) = skip_comment(state)?;
      state = new_state;
      comment = new_comment;
      let (new_state, new_spaces) = skip_spaces(state)?;
      state = new_state;
      spaces = new_spaces;
      if !comment && !spaces {
        return Ok((state, true));
      }
    }
  }
  Ok((state, false))
}

pub fn skip_parser<'a>() -> Parser<'a, bool> {
  Box::new(skip)
}

// Strings
// =======

/// Attempts to match a string right after the cursor.
/// Returns `true` if successful. Consumes string.
pub fn text_here<'a>(pat: &str, state: State<'a>) -> Answer<'a, bool> {
  if let Some(rest) = state.rest() {
    if rest.starts_with(pat) {
      let state = State { code: state.code, index: state.index + pat.len() };
      return Ok((state, true));
    }
  }
  Ok((state, false))
}

pub fn text_here_parser<'a>(pat: &'static str) -> Parser<'a, bool> {
  Box::new(move |x| text_here(pat, x))
}

/// Like 'text_here', but skips whitespace and comments first.
pub fn text<'a>(pat: &str, state: State<'a>) -> Answer<'a, bool> {
  let (state, _) = skip(state)?;
  let (state, matched) = text_here(pat, state)?;
  Ok((state, matched))
}

pub fn text_parser<'a>(pat: &'static str) -> Parser<'a, bool> {
  Box::new(move |x| text(pat, x))
}

/// Like 'text', but aborts if there is no match.
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

/// Returns `true` if cursor will be at the end of the file after skipping whitespace and comments.
pub fn done(state: State) -> Answer<bool> {
  let (state, _) = skip(state)?;
  Ok((state, state.index == state.code.len()))
}

pub fn done_parser<'a>() -> Parser<'a, bool> {
  Box::new(done)
}

// Blocks
// ======

/// Checks if a dry-run of the first parser returns `true`.
/// If so, applies the second parser and returns `Some`.
/// If no, returns `None`.
pub fn guard<'a, A: 'a>(
  head: Parser<'a, bool>,
  body: Parser<'a, A>,
  state: State<'a>,
) -> Answer<'a, Option<A>> {
  let (state, _) = skip(state)?;
  let (_, matched) = head(state)?;
  if matched {
    let (state, got) = body(state)?;
    Ok((state, Some(got)))
  } else {
    Ok((state, None))
  }
}

/// Applies optional parsers in sequence.
/// Returns the first that succeeds.
/// Aborts if none succeed.
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

pub fn maybe<'a, A: 'a>(parser: Parser<'a, A>, state: State<'a>) -> Answer<'a, Option<A>> {
  let result = parser(state);
  match result {
    Ok((state, result)) => Ok((state, Some(result))),
    Err(_) => Ok((state, None)),
  }
}

/// Evaluates a parser and returns its result, but reverts its effect.
pub fn dry<'a, A: 'a>(parser: Parser<'a, A>, state: State<'a>) -> Answer<'a, A> {
  let (_, result) = parser(state)?;
  Ok((state, result))
}

/// Evaluates a parser until a condition is met. Returns an array of results.
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

/// Evaluates a list-like parser, with an opener, separator, and closer.
pub fn list<'a, A: 'a, B: 'a>(
  parse_open: Parser<'a, bool>,
  parse_sep: Parser<'a, bool>,
  parse_close: Parser<'a, bool>,
  parse_elem: Parser<'a, A>,
  make: Box<dyn Fn(Vec<A>) -> B>,
  state: State<'a>,
) -> Answer<'a, B> {
  let (state, _) = parse_open(state)?;
  let mut state = state;
  let mut elems = Vec::new();
  loop {
    let (new_state, done) = parse_close(state)?;
    let (new_state, _) = parse_sep(new_state)?;
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

/// Checks if input is a valid character for names.
fn is_letter(chr: char) -> bool {
  chr.is_ascii_alphanumeric() || chr == '_' || chr == '.'
}

/// Parses a name right after the parsing cursor.
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

/// Parses a name after skipping comments and whitespace.
pub fn name(state: State) -> Answer<String> {
  let (state, _) = skip(state)?;
  name_here(state)
}

/// Parses a non-empty name after skipping.
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
  Err(format!("Expected {}:\n{}", name, &highlight(state.index, state.index + size, state.code)))
}

// WARN: This fails if `from_index` or `to_index` are not `char` boundaries.
// Should probably not use slice indexing directly, maybe use `get` method and
// handle possible error instead?
pub fn highlight(from_index: usize, to_index: usize, code: &str) -> String {
  debug_assert!(to_index >= from_index);
  debug_assert!(code.get(from_index..to_index).is_some());
  //let open = "<<<<####";
  //let close = "####>>>>";
  let open = "««««";
  let close = "»»»»";
  let open_color = "\x1b[4m\x1b[31m";
  let close_color = "\x1b[0m";
  let mut from_line = 0;
  let mut to_line = 0;
  for (i, c) in
    code.chars().enumerate().filter(|(_, c)| c == &'\n').take_while(|(i, _)| i < &to_index)
  {
    if i < from_index {
      from_line += c.len_utf8();
    }
    to_line += c.len_utf8();
  }
  let code =
    [&code[0..from_index], open, &code[from_index..to_index], close, &code[to_index..code.len()]]
      .concat();
  let block_from_line = std::cmp::max(from_line as i64 - 3, 0) as usize;
  let block_to_line = std::cmp::min(to_line + 3, code.lines().count());
  code
    .lines()
    .enumerate()
    .skip_while(|(i, _)| i < &block_from_line)
    .take_while(|(i, _)| i < &block_to_line)
    .map(|(_, line)| line)
    .enumerate()
    .format_with("", |(i, line), f| {
      let numb = block_from_line + i;
      // TODO: An allocation of an intermediate string still occurs here
      // which is inefficient. Should figure out how to improve this.
      let rest = if numb == from_line && numb == to_line {
        [
          &line[0..find(line, open)],
          open_color,
          &line[find(line, open) + open.len()..find(line, close)],
          close_color,
          &line[find(line, close) + close.len()..line.len()],
          "\n",
        ]
        .concat()
      } else if numb == from_line {
        [&line[0..find(line, open)], open_color, &line[find(line, open)..line.len()], "\n"].concat()
      } else if numb > from_line && numb < to_line {
        [open_color, line, close_color, "\n"].concat()
      } else if numb == to_line {
        [
          &line[0..find(line, open)],
          open_color,
          &line[find(line, open)..find(line, close) + close.len()],
          close_color,
          "\n",
        ]
        .concat()
      } else {
        [line, "\n"].concat()
      };
      f(&format_args!("    {} | {}", numb, rest))
    })
    .to_string()
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
        let (state, _) = consume("(", state)?;
        let (state, lft) = testree_parser()(state)?;
        let (state, rgt) = testree_parser()(state)?;
        let (state, _) = consume(")", state)?;
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

#[cfg(test)]
mod tests {
  use super::*;
  use proptest::prelude::*;

  mod old_parser {
    use super::super::*;

    pub fn flatten(texts: &[&str]) -> String {
      texts.concat()
    }

    pub fn lines(text: &str) -> Vec<String> {
      text.lines().map(String::from).collect()
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
  }

  // Matches anything.
  const RE_ANY: &str = "(?s).*";

  #[derive(Debug)]
  struct MockState {
    code: String,
    index: usize,
  }

  prop_compose! {
    fn state_tail()(
      any in RE_ANY, ch in any::<char>()
    ) -> MockState {
      let code = format!("{}{}", ch, any);
      let index = ch.len_utf8();
      MockState { code, index }
    }
  }

  proptest! {
    #[test]
    fn test_tail(state in state_tail()) {
      let state_after = tail(State {
        code: &state.code, index: 0
      });
      prop_assert_eq!(state.index, state_after.index);
      prop_assert!(
        state_after.index <= state.code.len(),
        "\ncode length: {}\nindex: {}\n",
        state.code.len(),
        state_after.index
      );
    }
  }

  const COMMENT: &str = "//";
  // Matches any line (i.e., that doesn't contain `'\r'` or `'\n'`).
  const RE_LINE: &str = "[^\r\n]*";

  prop_compose! {
    fn state_skip_comment()(
      line in RE_LINE.prop_filter(
        "Values must not start with `COMMENT`.", |a| !a.starts_with(COMMENT)),
      will_comment in any::<bool>(),
      any in RE_ANY,
    ) -> (MockState, bool) {
      let index = if will_comment {
        COMMENT.len() + line.len()
      } else {
        0
      };
      let code = if will_comment {
        format!("{}{}\n{}", COMMENT, line, any)
      } else {
        format!("{}\n{}", line, any)
      };
      (MockState { code, index }, will_comment)
    }
  }

  proptest! {
    #[test]
    fn test_skip_comment(state in state_skip_comment()) {
      let answer = skip_comment(State {
        code: &state.0.code, index: 0
      }).unwrap();
      let state_after = answer.0;
      prop_assert_eq!(state.0.index, state_after.index);
      prop_assert_eq!(state.1, answer.1);
      prop_assert!(
        state_after.index <= state.0.code.len(),
        "\ncode length: {}\nindex: {}\n",
        state.0.code.len(),
        state_after.index
      );
    }
  }

  const RE_WHITESPACE: &str = "\\s+";

  prop_compose! {
    fn state_skip_spaces()(
      any in RE_ANY.prop_filter(
        "Values must not start with whitespace.", |a| a == a.trim_start()),
      has_spaces in any::<bool>(),
      spaces in RE_WHITESPACE,
    ) -> (MockState, bool) {
      let index = if has_spaces {
        spaces.len()
      } else {
        0
      };
      let code = if has_spaces {
        format!("{}{}", spaces, any)
      } else {
        any
      };
      (MockState { code, index }, has_spaces)
    }
  }

  proptest! {
    #[test]
    fn test_skip_spaces(state in state_skip_spaces()) {
      let answer = skip_spaces(State {
        code: &state.0.code, index: 0
      }).unwrap();
      let state_after = answer.0;
      prop_assert_eq!(state.0.index, state_after.index);
      prop_assert_eq!(state.1, answer.1);
      prop_assert!(
        state_after.index <= state.0.code.len(),
        "\ncode length: {}\nindex: {}\n",
        state.0.code.len(),
        state_after.index
      );
    }
  }

  prop_compose! {
    fn state_skip()(
      will_skip in any::<bool>(),
      any in RE_ANY.prop_filter(
        "Values must not start with whitespace or be a comment.", |a| {
        let a_trimmed = a.trim_start();
        !a_trimmed.starts_with(COMMENT) && a == a_trimmed
      }),
    )(
      spaces_comments in if will_skip {
        prop::collection::vec((RE_WHITESPACE, RE_LINE), 0..10)
      } else {
        prop::collection::vec(("", ""), 0)
      },
      will_skip in Just(will_skip),
      any in Just(any),
    ) -> (MockState, bool) {
      let mut code: String = if will_skip {
        spaces_comments
          .iter()
          .flat_map(|(space, comment)|
            [space.as_str(), COMMENT, comment.as_str(), "\n"])
          .collect()
      } else {
        String::with_capacity(any.len())
      };
      let index = code.len();
      code.push_str(&any);
      let will_skip = code.trim_start() != code || code.starts_with(COMMENT);
      (MockState { code, index }, will_skip)
    }
  }

  proptest! {
    #[test]
    fn test_skip(state in state_skip()) {
      let answer = skip(State {
        code: &state.0.code, index: 0
      }).unwrap();
      let state_after = answer.0;
      prop_assert_eq!(state.0.index, state_after.index);
      prop_assert_eq!(state.1, answer.1);
      prop_assert!(
        state_after.index <= state.0.code.len(),
        "\ncode length: {}\nindex: {}\n",
        state.0.code.len(),
        state_after.index
      );
    }
  }

  prop_compose! {
    fn range(from: usize, to: usize)(from in from..to)(
      to in from..to,
      from in Just(from)
    ) -> (usize, usize) {
      (from, to)
    }
  }

  // Matches lines with at least a single character.
  const RE_NON_EMPTY: &str = ".{1,}";

  prop_compose! {
    fn args_highlight()(code in RE_NON_EMPTY)(
      code in Just(code.clone()),
      (from, to) in range(0, code.len()).prop_filter(
        "Values must be `char` boundaries.", move |(from, to)| {
        code.is_char_boundary(*from) && code.is_char_boundary(*to)
      })
    ) -> (usize, usize, String) {
      (from, to, code)
    }
  }

  proptest! {
    #[test]
    fn test_highlight((from_index, to_index, code) in args_highlight()) {
      prop_assert_eq!(
        old_parser::highlight(from_index, to_index, &code),
        highlight(from_index, to_index, &code)
      );
    }
  }
}
