use crate::text::*;

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
struct State<'a> {
  code: &'a Text,
  index: usize,
}

type Parser<A> = Box<dyn FnOnce(State) -> (State, A)>;

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
  return Box::new(move |state| {
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

fn matchs(match_code: &'static Text) -> Parser<bool> {
  return Box::new(move |state| {
    let (state, skipped) = skip(state);
    return match_here(match_code)(state);
  });
}

fn consume(c: &'static Text) -> Parser<()> {
  return Box::new(move |state| {
    let (state, matched) = match_here(c)(state);
    if matched {
      return (state, ());
    } else {
      return expected_string(c)(state);
    }
  });
}

fn get_char() -> Parser<char> {
  return Box::new(move |state| {
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

fn done() -> Parser<bool> {
  return Box::new(move |state| {
    let (state, skipped) = skip(state);
    return (state, state.index == state.code.len());
  });
}

fn guard<A: 'static>(head: Parser<bool>, body: Parser<A>) -> Parser<Option<A>> {
  return Box::new(move |state| {
    let (state, skipped) = skip(state);
    let (state, matched) = dry(head)(state);
    if matched {
      let (state, got) = body(state);
      return (state, Some(got));
    } else {
      return (state, None);
    }
  });
}

fn grammar<A: 'static>(name: &'static Text, choices: Vec<Parser<Option<A>>>) -> Parser<A> {
  return Box::new(move |state| {
    //for i in 0..choices.len() {
    for choice in choices {
      let (state, got) = choice(state);
      if got.is_some() {
        return (state, got.unwrap());
      }
    }
    return expected_type(name)(state);
  });
}

fn dry<A: 'static>(parser: Parser<A>) -> Parser<A> {
  return Box::new(move |state| {
    let (state, result) = parser(state);
    return (state, result);
  });
}

fn expected_string<A>(c: &'static Text) -> Parser<A> {
  return Box::new(move |state| {
    panic!(
      "Expected '{}':\n{}",
      "TODO_text_to_utf8", "TODO_HIGHLIGHT_FUNCTION"
    );
  });
}

fn expected_type<A>(name: &'static Text) -> Parser<A> {
  return Box::new(move |state| {
    panic!(
      "Expected {}:\n{}",
      "TODO_text_to_utf8", "TODO_HIGHLIGHT_FUNCTION"
    );
  });
}
