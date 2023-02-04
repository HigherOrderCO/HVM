// This parse library is more highoer-order and functional than existing alternatives.
// It explores the `?` syntax of `Result` to have a do-block-like monadic notation.
// A Parser is defined as (with details omitted):
//
//   Answer<A> = Result<(State, A), String>
//   Parser<A> = Fn(State) -> Answer<A>>
//
// There are 2 ways to fail:
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

pub fn read<'a, A>(parser: Parser<'a, A>, code: &'a str) -> Result<A, String> {
  match parser(State { code, index: 0 }) {
    Ok((_, value)) => Ok(value),
    Err(msg) => Err(msg),
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

// Heads
// =====

/// Returns and consumes the next char, or '\0'.
pub fn here_take_head(state: State) -> Answer<char> {
  if let Some(got) = head(state) {
    let state = State { code: state.code, index: state.index + got.len_utf8() };
    Ok((state, got))
  } else {
    Ok((state, '\0'))
  }
}

/// Monadic here_take_head.
pub fn do_here_take_head<'a>() -> Parser<'a, char> {
  Box::new(here_take_head)
}

/// Skip, then here_take_head.
pub fn there_take_head(state: State) -> Answer<char> {
  let (state, _) = skip(state)?;
  here_take_head(state)
}

/// Monadic there_take_head.
pub fn do_there_take_head<'a>() -> Parser<'a, char> {
  Box::new(there_take_head)
}

/// Returns, without consuming, the next char, or '\0'.
pub fn here_peek_head(state: State) -> Answer<char> {
  if let Some(got) = head(state) {
    return Ok((state, got));
  } else {
    return Ok((state, '\0'));
  }
}

/// Monadic here_peek_head.
pub fn do_here_peek_head<'a>() -> Parser<'a, char> {
  Box::new(here_peek_head)
}

/// Skip, then here_peek_head.
pub fn there_peek_head(state: State) -> Answer<char> {
  let (state, _) = skip(state)?;
  here_peek_head(state)
}

/// Monadic peek_symol.
pub fn do_there_peek_head<'a>() -> Parser<'a, char> {
  Box::new(there_peek_head)
}

// Skippers
// ========

/// Skips characters until a condition is false.
pub fn skip_while(mut state: State, cond: Box<dyn Fn(&char) -> bool>) -> Answer<bool> {
  if let Some(rest) = state.rest() {
    let add: usize = rest.chars().take_while(cond).map(|a| a.len_utf8()).sum();
    state.index += add;
    if add > 0 {
      return Ok((state, true));
    }
  }
  Ok((state, false))
}

/// Skips C-like comments, starting with '//'.
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

/// Monadic skip_comment.
pub fn do_skip_comment<'a>() -> Parser<'a, bool> {
  Box::new(skip_comment)
}

/// Skips whitespaces.
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

/// Monadic skip_spaces.
pub fn do_skip_spaces<'a>() -> Parser<'a, bool> {
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

/// Monadic skip.
pub fn do_skip<'a>() -> Parser<'a, bool> {
  Box::new(skip)
}

// Exacts
// ======

/// Attempts to match a string right after the cursor.
/// Returns `true` if successful. Consumes string.
pub fn here_take_exact<'a>(pat: &str, state: State<'a>) -> Answer<'a, bool> {
  if let Some(rest) = state.rest() {
    if rest.starts_with(pat) {
      let state = State { code: state.code, index: state.index + pat.len() };
      return Ok((state, true));
    }
  }
  Ok((state, false))
}

/// Monadic 'here_take_exact'.
pub fn do_here_take_exact<'a>(pat: &'static str) -> Parser<'a, bool> {
  Box::new(move |x| here_take_exact(pat, x))
}

/// Skip, then 'here_take_exact'.
pub fn there_take_exact<'a>(pat: &str, state: State<'a>) -> Answer<'a, bool> {
  let (state, _) = skip(state)?;
  let (state, matched) = here_take_exact(pat, state)?;
  Ok((state, matched))
}

/// Monadic 'there_take_exact'.
pub fn do_there_take_exact<'a>(pat: &'static str) -> Parser<'a, bool> {
  Box::new(move |x| there_take_exact(pat, x))
}

/// Attempts to match a string right after the cursor.
/// Returns `true` if successful. Doesn't consume it.
pub fn here_peek_exact<'a>(pat: &str, state: State<'a>) -> Answer<'a, bool> {
  if let Some(rest) = state.rest() {
    if rest.starts_with(pat) {
      return Ok((state, true));
    }
  }
  Ok((state, false))
}

/// Monadic 'here_take_exact'.
pub fn do_here_peek_exact<'a>(pat: &'static str) -> Parser<'a, bool> {
  Box::new(move |x| here_peek_exact(pat, x))
}

/// Perform `there_take_exact`. Abort if not possible.
pub fn force_there_take_exact<'a>(pat: &str, state: State<'a>) -> Answer<'a, ()> {
  let (state, matched) = there_take_exact(pat, state)?;
  if matched {
    Ok((state, ()))
  } else {
    expected(pat, pat.len(), state)
  }
}

/// Skip, then 'here_peek_exact'.
pub fn there_peek_exact<'a>(pat: &str, state: State<'a>) -> Answer<'a, bool> {
  let (state, _) = skip(state)?;
  let (state, matched) = here_peek_exact(pat, state)?;
  Ok((state, matched))
}

/// Monadic 'there_peek_exact'.
pub fn do_there_peek_exact<'a>(pat: &'static str) -> Parser<'a, bool> {
  Box::new(move |x| there_peek_exact(pat, x))
}

/// Monadic `force_there_take_exact`.
pub fn do_force_there_take_exact<'a>(pat: &'static str) -> Parser<'a, ()> {
  Box::new(move |x| force_there_take_exact(pat, x))
}

// End
// ===

/// Returns `true` if cursor will be at the end of the file.
pub fn here_end(state: State) -> Answer<bool> {
  Ok((state, state.index == state.code.len()))
}

/// Monadic `here_end`.
pub fn do_here_end<'a>() -> Parser<'a, bool> {
  Box::new(there_end)
}

/// Skip, then `here_end`.
pub fn there_end(state: State) -> Answer<bool> {
  let (state, _) = skip(state)?;
  Ok((state, state.index == state.code.len()))
}

/// Monadic `there_end`.
pub fn do_there_end<'a>() -> Parser<'a, bool> {
  Box::new(there_end)
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

/// Attempts several parsers.
/// Returns true if at least one succeeds.
pub fn any<'a>(parsers: &[Parser<'a, bool>], state: State<'a>) -> Answer<'a, bool> {
  for parser in parsers {
    let (state, matched) = parser(state)?;
    if matched {
      return Ok((state, true));
    }
  }
  Ok((state, false))
}

/// Applies optional parsers in sequence.
/// Returns the first that succeeds.
/// If none succeeds, aborts.
pub fn attempt<'a, A: 'a>(
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

/// Takes a parser that can abort, and transforms it into a non abortive parser
/// that returns Some if it succeeds, and None if it fails.
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
  chr.is_ascii_alphanumeric() || chr == '_' || chr == '.' || chr == '$'
}

/// Parses a name right after the parsing cursor.
pub fn here_name(state: State) -> Answer<String> {
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
pub fn there_name(state: State) -> Answer<String> {
  let (state, _) = skip(state)?;
  here_name(state)
}

/// Parses a non-empty name after skipping.
pub fn there_nonempty_name(state: State) -> Answer<String> {
  let (state, name1) = there_name(state)?;
  if !name1.is_empty() {
    Ok((state, name1))
  } else {
    expected("name", 1, state)
  }
}

// Errors
// ======

pub fn expected<'a, A>(name: &str, size: usize, state: State<'a>) -> Answer<'a, A> {
  Err(format!("Expected `{}`:\n{}", name, &highlight_error::highlight_error(state.index, state.index + size, state.code)))
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
      do_there_take_exact("("),
      Box::new(|state| {
        let (state, _) = force_there_take_exact("(", state)?;
        let (state, lft) = testree_parser()(state)?;
        let (state, rgt) = testree_parser()(state)?;
        let (state, _) = force_there_take_exact(")", state)?;
        Ok((state, Box::new(Testree::Node { lft, rgt })))
      }),
      state,
    )
  })
}

pub fn leaf_parser<'a>() -> Parser<'a, Option<Box<Testree>>> {
  Box::new(|state| {
    guard(
      do_there_take_exact(""),
      Box::new(|state| {
        let (state, val) = there_name(state)?;
        Ok((state, Box::new(Testree::Leaf { val })))
      }),
      state,
    )
  })
}

pub fn testree_parser<'a>() -> Parser<'a, Box<Testree>> {
  Box::new(|state| {
    let (state, tree) = attempt("Testree", &[node_parser(), leaf_parser()], state)?;
    Ok((state, tree))
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use proptest::prelude::*;

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
          .flat_map(|(space, comment)| [space.as_str(), COMMENT, comment.as_str(), "\n"])
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

  // proptest! {
  //   #[test]
  //   fn test_highlight((from_index, to_index, code) in args_highlight()) {
  //     prop_assert_eq!(
  //       old_parser::highlight(from_index, to_index, &code),
  //       highlight(from_index, to_index, &code)
  //     );
  //   }
  // }
}
