#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

//use std::time::{SystemTime, UNIX_EPOCH};
use crate::parser::*;
use crate::text::Text;
use std::rc::Rc;
use std::time::Instant;

mod lambolt;
mod parser;
mod readback;
mod runtime;
mod text;

use runtime as rt;

enum StrBinTree {
  Tie {
    left: Box<StrBinTree>,
    right: Box<StrBinTree>,
  },
  Tip, //{val: Text}
}

fn sbt_tip(state: State) -> (State, Option<StrBinTree>) {
  //  println!("sbt_tip");
  (state, Some(StrBinTree::Tip))
}
fn sbt_tie(state: State) -> (State, Option<StrBinTree>) {
  let (state, left) = sbt(state);
  match left {
    Some(left_val) => {
      let (state, right) = sbt(state);
      match right {
        Some(right_val) => (
          state,
          Some(StrBinTree::Tie {
            left: Box::new(left_val),
            right: Box::new(right_val),
          }),
        ),
        None => (state, None),
      }
    }
    None => (state, None),
  }
}
fn sbt(state: State) -> (State, Option<StrBinTree>) {
  //    println!("sbt:");
  //    debug(state);
  //    println!("");
  let left_paren = Rc::new(text::utf8_to_text("("));
  let right_paren = Rc::new(text::utf8_to_text(")"));
  let (state, has_left_paren) = matchs(left_paren)(state);
  if has_left_paren {
    let (state, result) = grammar(
      &text::utf8_to_text("bintree"),
      vec![Rc::new(sbt_tie), Rc::new(sbt_tip)],
    )(state);
    match result {
      Some(tree) => {
        let (state, has_right_paren) = matchs(right_paren)(state);
        if has_right_paren {
          //            println!("sbt_tie");
          (state, Some(tree))
        } else {
          (state, None)
        }
      }
      None => (state, None),
    }
  } else {
    (state, None)
  }
}
fn main() {
  let state = parser::State {
    code: &text::utf8_to_text("((()(()()))())"),
    index: 0,
  };
  let (state, result) = sbt(state);
  match result {
    Some(tree) => println!("parsed"),
    None => println!("not parsed"),
  };
  //  if result {
  //      println!("it's true, while the awful pray for tomorrow");
  //  } else {
  //      println!("it's false");
  //  }

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
