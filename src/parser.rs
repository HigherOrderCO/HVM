use crate::text::*;
use std::rc::Rc;

// Regex
// -----

// Text
// ----

//TypeScript
//export type State = {
//  code: string,
//  index: number
//};
//export type Parser<A> = (state: State) => [State, A];
//Rust:
#[derive(Clone, Copy, Debug)]
pub struct State<'a> {
  pub code: &'a Text,
  pub index: usize,
}

pub type Parser<'a, A> = Rc<dyn Fn(State) -> (State, A) + 'a>;

fn read<A>(parser: fn(state: State) -> (State, A), code: &Text) -> A {
  let (state, value) = parser(State { code, index: 0 });
  return value;
}

fn skip_comment(mut state: State) -> (State, bool) {
  let skips = equal_at(&state.code, &vec!['/', '/'], state.index);
  if skips {
    state.index += 2;
    while state.index < state.code.len() && equal_at(&state.code, &vec!['\n'], state.index) {
      state.index += 1;
    }
  }
  return (state, skips);
}

fn skip_spaces(mut state: State) -> (State, bool) {
  let mut skips = equal_at(&state.code, &vec![' '], state.index);
  while skips {
    state.index += 1;
    skips = equal_at(&state.code, &vec![' '], state.index);
  }
  return (state, skips);
}

fn skip(state: State) -> (State, bool) {
  let (state, comment) = skip_comment(state);
  let (state, spaces) = skip_spaces(state);
  if comment || spaces {
    let (state, skipped) = skip(state);
    return (state, true);
  } else {
    return (state, false);
  }
}

fn match_here(c: &'static Text) -> Parser<bool> {
  return Rc::new(move |state| {
    if equal_at(&state.code, c, state.index) {
      return (
        State {
          code: state.code,
          index: state.index + c.len(),
        },
        true,
      );
    } else {
      return (state, false);
    }
  });
}

fn until<'a, A: 'a>(delim: Parser<'a, bool>, parser: Parser<'a, A>) -> Parser<'a, Vec<A>> {
  Rc::new(move |state| {
    let mut ret = Vec::new();
    let mut delimited = true;
    let mut state = state;
    while delimited {
      let (new_state, new_delimited) = delim(state);
      if new_delimited {
        let (new_state, parsed) = parser(new_state);
        ret.push(parsed);
        state = new_state;
      } else {
        state = new_state;
      }
      delimited = new_delimited;
    }
    (state, ret)
  })
}

fn matchs<'a>(match_code: &'static Text) -> Parser<'a, bool> {
  return Rc::new(move |state| {
    let (state, skipped) = skip(state);
    return match_here(match_code)(state);
  });
}

fn consume(c: &'static Text) -> Parser<()> {
  return Rc::new(move |state| {
    let (state, matched) = match_here(c)(state);
    if matched {
      return (state, ());
    } else {
      return expected_string(c)(state);
    }
  });
}

fn get_char<'a>() -> Parser<'a, char> {
  return Rc::new(move |state| {
    let (state, skipped) = skip(state);
    if state.index < state.code.len() {
      return (
        State {
          code: state.code,
          index: state.index + 1,
        },
        state.code[state.index],
      );
    } else {
      return (state, '\0');
    }
  });
}

fn done<'a>() -> Parser<'a, bool> {
  return Rc::new(move |state| {
    let (state, skipped) = skip(state);
    return (state, state.index == state.code.len());
  });
}

fn guard<'a, A: 'a>(head: Parser<'a, bool>, body: Parser<'a, A>) -> Parser<'a, Option<A>> {
  return Rc::new(move |state| {
    let (state, skipped) = skip(state);
    let (state, matched) = dry(head.clone())(state);
    if matched {
      let (state, got) = body(state);
      return (state, Some(got));
    } else {
      return (state, None);
    }
  });
}

fn grammar<'a, A: 'a>(name: &'static Text, choices: Vec<Parser<'a, Option<A>>>) -> Parser<'a, A> {
  return Rc::new(move |state| {
    //for i in 0..choices.len() {
    for choice in &choices {
      let (state, got) = choice(state);
      if got.is_some() {
        return (state, got.unwrap());
      }
    }
    return expected_type(name)(state);
  });
}

fn dry<'a, A: 'a>(parser: Parser<'a, A>) -> Parser<'a, A> {
  return Rc::new(move |state| {
    let (state, result) = parser(state);
    return (state, result);
  });
}

fn expected_string<A>(c: &'static Text) -> Parser<A> {
  return Rc::new(move |state| {
    panic!(
      "Expected '{}':\n{}",
      "TODO_text_to_utf8", "TODO_HIGHLIGHT_FUNCTION"
    );
  });
}

fn expected_type<A>(name: &'static Text) -> Parser<A> {
  return Rc::new(move |state| {
    panic!(
      "Expected {}:\n{}",
      "TODO_text_to_utf8", "TODO_HIGHLIGHT_FUNCTION"
    );
  });
}

// Evaluates a list-like parser, with an opener, separator, and closer.
fn list<'a, A: 'a, B: 'a>(
  open: Parser<'a, bool>,
  sep: Parser<'a, bool>,
  close: Parser<'a, bool>,
  elem: Parser<'a, A>,
  make: fn(x: Vec<A>) -> B,
) -> Parser<'a, B> {
  return Rc::new(move |state| {
    let (state, skp) = open(state);
    let (state, arr) = until(
      close.clone(),
      Rc::new(|state| {
        let (state, val) = elem(state);
        let (state, skp) = sep(state);
        (state, val)
      }),
    )(state);
    (state, make(arr))
  });
}
