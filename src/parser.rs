//struct State {
  //code: String,
  //index: u64
//}

//type Parse<A> = fn(state: State) -> (State, A);

//fn read<A>(parser: fn(state: State) -> (State, A), code: String) -> A {
  //let (state, value) = parser(State {code, index: 0});
  //return value;
//}

//fn skip_comment(state: State) -> (State, bool) {
  //let state = State {code: state.code, index: state.index};
  //let skips = &(&state.code)[state.index as usize .. (state.index + 2) as usize] == "//";
  //if skips {
    //state.index += 2;
    //while state.index < state.code.len() && !state.code[state.index] != '\n' {
      //state.index += 1;
    //}
  //}
  //return (state, skips);
//}
