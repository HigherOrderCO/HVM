type Code = Vec<char>;

//TypeScript
//export type State = {
//  code: string,
//  index: number
//};
//export type Parser<A> = (state: State) => [State, A];
//Rust:
#[derive(Clone, Copy, Debug)]
struct State<'a> {
  code: &'a Code,
  index: usize
}

type Parser<A> = Box<dyn FnOnce(State) -> (State, A)>;

// TypeScript:
//export function read<A>(parser: () => Parser<A>, code: string): A {
  //var [state, value] = parser()({code, index: 0});
  //return value;
//}
// Rust:
fn read<A>(parser: fn(state: State) -> (State, A), code: &Code) -> A {
  let (state, value) = parser(State {code, index: 0});
  return value;
}

fn equal_at(code: &Code, test: &Code, i: usize) -> bool {
  for j in 0..test.len() {
    if test[i as usize + j] != test[j as usize] {
      return false;
    }
  }
  return true;
}

// TypeScript: 
//export const skip_comment : Parser<boolean> = (state: State) => {
  //var state = {...state};
  //var skips = state.code.slice(state.index, state.index + 2) === "//";
  //if (skips) {
    //state.index += 2;
    //while (state.index < state.code.length && !/\n/.test(state.code[state.index])) {
      //state.index += 1;
    //}
  //}
  //return [state, skips];
//};
// Rust:
fn skip_comment(mut state: State) -> (State, bool) {
  let skips = equal_at(&state.code, &vec!['/','/'], state.index);
  if skips {
    state.index += 2;
    while state.index < state.code.len() && equal_at(&state.code, &vec!['\n'], state.index) {
      state.index += 1;
    }
  }
  return (state, skips);
}

// TypeScript:
//export const skip_spaces : Parser<boolean> = (state: State) => {
  //var state = {...state};
  //var skips = /\s/.test(state.code[state.index]);
  //while (/\s/.test(state.code[state.index])) {
    //state.index += 1;
  //}
  //return [state, skips];
//};
// Rust:
fn skip_spaces(mut state: State) -> (State, bool) {
  let mut skips = equal_at(&state.code, &vec![' '], state.index);
  while skips {
    state.index += 1;
    skips = equal_at(&state.code, &vec![' '], state.index);
  }
  return (state, skips);
}

// TypeScript:
//export const skip : Parser<boolean> = (state: State) => {
  //var [state, comment] = skip_comment(state);
  //var [state, spaces] = skip_spaces(state);
  //if (comment || spaces) {
    //var [state, skipped] = skip(state);
    //return [state, true];
  //} else {
    //return [state, false];
  //}
//};
// Rust:
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

// TypeScript:
//export function match_here(str: string) : Parser<boolean> {
  //return (state) => {
    //if (state.code.slice(state.index, state.index + str.length) === str) {
      //return [{...state, index: state.index + str.length}, true];
    //} else {
      //return [state, false];
    //}
  //};
//}
// Rust:
fn match_here(c: &'static Code) -> Parser<bool> {
  return Box::new(move |state| {
    if equal_at(&state.code, c, state.index) {
      return (State {code: state.code, index: state.index + c.len()}, true);
    } else {
      return (state, false);
    }
  });
}

// TypeScript:
//export function match(matcher: string | RegExp) : Parser<boolean> {
  //return (state) => {
    //var [state, skipped] = skip(state);
    //return match_here(matcher)(state);
  //};
//}
// Rust:
fn matchs(match_code: &'static Code) -> Parser<bool> {
  return Box::new(move |state| {
    let (state, skipped) = skip(state);
    return match_here(match_code)(state);
  });
}

// TypeScript:
//export function consume(str: string) : Parser<null> {
  //return (state) => {
    //var [state, matched] = match(str)(state);
    //if (matched) {
      //return [state, null];
    //} else {
      //var fail : Parser<null> = expected_string(str);
      //return fail(state);
    //}
  //};
//}
// Rust:
fn consume(c: &'static Code) -> Parser<()> {
  return Box::new(move |state| {
    let (state, matched) = match_here(c)(state);
    if matched {
      return (state, ());
    } else {
      return expected_string(c)(state);
    }
  });
}

// TypeScript:
//export function get_char(): Parser<string | null> {
  //return (state) => {
    //var [state, skipped] = skip(state);
    //if (state.index < state.code.length) {
      //return [{...state, index: state.index + 1}, state.code[state.index]];
    //} else {
      //return [state, null];
    //}
  //}
//}
// Rust:
fn get_char() -> Parser<char> {
  return Box::new(move |state| {
    let (state, skipped) = skip(state);
    if state.index < state.code.len() {
      return (State {code: state.code, index: state.index + 1}, state.code[state.index]);
    } else {
      return (state, '\0');
    }
  });
}

// TypeScript:
//export const done : Parser<boolean> = (state) => {
  //var [state, skipped] = skip(state);
  //return [state, state.index === state.code.length];
//};
// Rust:
fn done() -> Parser<bool> {
  return Box::new(move |state| {
    let (state, skipped) = skip(state);
    return (state, state.index == state.code.len());
  });
}

// TypeScript:
//export function guard<A>(head: Parser<boolean>, body: Parser<A>) : Parser<A|null> {
  //return (state) => {
    //var [state, skipped] = skip(state);
    //var [state, matched] = dry(head)(state);
    //if (matched) {
      //return body(state);
    //} else {
      //return [state, null];
    //}
  //};
//}
// Rust:
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

// TypeScript:
//export function grammar<A>(name: string, choices: Array<Parser<A|null>>) : Parser<A> {
  //return (state) => {
    //for (var i = 0; i < choices.length; ++i) {
      //var [state, got] = choices[i](state);
      //if (got) {
        //return [state, got];
      //}
    //}
    //var fail : Parser<A> = expected_type(name);
    //return fail(state);
  //};
//}
// Rust:
//fn grammar<A: 'static>(name: &'static str, choices: Vec<Parser<Option<A>>>) -> Parser<A> {
  //return Box::new(move |state| {
    //for i in 0..choices.len() {
      //let (state, got) = choices[i](state);
      //if got.is_some() {
        //return (state, got.unwrap());
      //}
    //}
    //return expected_type(name)(state);
  //});
//}

// TypeScript:
// export function dry<A>(parser: Parser<A>): Parser<A> {
  //return (state) => {
    //var [_, result] = parser(state);
    //return [state, result];
  //};
//}
// Rust:
fn dry<A: 'static>(parser: Parser<A>) -> Parser<A> {
  return Box::new(move |state| {
    let (state, result) = parser(state);
    return (state, result);
  });
}

// TypeScript:
//export function expected_string<A>(str: string): Parser<A> {
  //return (state) => {
    //throw "Expected '" + str + "':\n" + highlight(state.index, state.index + str.length, state.code);
  //}
//}
// Rust:
fn expected_string<A>(c: &'static Code) -> Parser<A> {
  return Box::new(move |state| {
    panic!("Expected '{}':\n{}", "TODO_CODE_TO_STRING", "TODO_HIGHLIGHT_FUNCTION");
  });
}

// TypeScript:
//export function expected_type<A>(name: string): Parser<A> {
  //return (state) => {
    //throw "Expected " + name + ":\n" + highlight(state.index, state.index + 1, state.code);
  //}
//}
// Rust:
//fn expected_type<A>(name: &'static Code) -> Parser<A> {
  //return Box::new(move |state| {
    //panic!("Expected {}:\n{}", name, "TODO_HIGHLIGHT_FUNCTION");
  //});
//}
