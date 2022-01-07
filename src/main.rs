#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

//use std::time::{SystemTime, UNIX_EPOCH};
use std::time::Instant;
use std::rc::Rc;
use crate::text::Text;
use crate::parser::*;

mod lambolt;
mod parser;
mod readback;
mod runtime;
mod text;

use runtime as rt;

enum StrBinTree {
  Tie{left: Box<StrBinTree>, right: Box<StrBinTree>},
  Tip //{val: Text}
}

fn sbt<'a>(state: State) -> (State, bool) {
    let text = Rc::new(text::utf8_to_text(
        "()"
      ));
    let (state, either) = matchs(text)(state);
    (state, either)
}
fn main() {
  let state = parser::State {
      code: &text::utf8_to_text(
        "()"
      ),
      index: 0
  };
  let (new_state, result) = sbt(state);
  if result {
      println!("it's true, while the awful pray for tomorrow");
  } else {
      println!("it's false");
  }
  //let mut worker = rt::new_worker();
  //worker.size = 1;
  //worker.node[0] = rt::Cal(0, 0, 0);
  //let start = Instant::now();
  //rt::normal(&mut worker, 0);
  //let total = (start.elapsed().as_millis() as f64) / 1000.0;
  //println!("* rwts: {} ({:.2}m rwt/s)", worker.cost, (worker.cost as f64) / total / 1000000.0);
  //println!("* time: {:?}", total);

//  println!(
//    "{}",
//    text::text_to_utf8(&text::highlight(
//      3,
//      7,
//      &text::utf8_to_text(
//        "oi tudo bem? como vai você hoje?\neu pessoalmente estou ok.\nespero que vc tbm"
//      )
//    ))
//  );
//  println!(":pp");
}
