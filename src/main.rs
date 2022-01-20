#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_parens)]
#![allow(non_snake_case)]
#![allow(unused_imports)]

mod rulebook;
mod compiler;
mod dynfun;
mod lambolt;
mod parser;
mod readback;
mod runtime;

use std::time::Instant;

fn main() {
  let (norm, cost, time) = eval("Main", "
    (Slow (Z))      = 1
    (Slow (S pred)) = (+ (Slow pred) (Slow pred))

    (Main) = (Slow 
        (S(S(S (S(S(S(S
      (S(S(S(S (S(S(S(S
      (S(S(S(S (S(S(S(S
      (Z)
      )))) ))))
      )))) ))))
      )))) )))
    )
  ");

  println!("{}", norm);
  println!("- rwts: {} ({:.2} rwt/s)", cost, (cost as f64) / (time as f64));
}

use dynfun as df;
use runtime as rt;

// Evaluates a Lambolt term to normal form
fn eval(main: &str, code: &str) -> (String, u64, u64) {
  // Creates a new Runtime worker
  let mut worker = runtime::new_worker();

  // Parses and reads the input file
  let file = lambolt::read_file(code);

  // Converts the Lambolt file to a rulebook file
  let book = rulebook::gen_rulebook(&file);

  // Builds dynamic functions
  let funs = dynfun::build_runtime_functions(&book);

  // FIXME: I'm using this to optimize dynfuns! Remove later.
  //funs.insert(0, hardcoded_slow_function());

  // Builds a runtime "(Main)" term
  let main = lambolt::read_term("(Main)");
  let host = dynfun::alloc_term(&mut worker, &book, &main);

  // Normalizes it
  let init = Instant::now();
  runtime::normal(&mut worker, host, &funs, Some(&book.id_to_name));
  let time = init.elapsed().as_millis() as u64;

  // Reads it back to a Lambolt string
  let norm = readback::as_code(&worker, &Some(book), host);

  // Returns the normal form and the gas cost
  (norm, worker.cost, time)
}

// FIXME: I'm using this to optimize dynfuns. Remove later!
fn hardcoded_slow_function() -> rt::Function {
  let dynfun = df::DynFun {
    redex: vec![true],
    rules: vec![
      df::DynRule {
        cond: vec![0x8000000100000000],
        vars: vec![],
        body: df::DynTerm::U32 { numb: 1 },
        free: vec![(0, 0)]
      },
      df::DynRule {
        cond: vec![0x8100000200000000],
        vars: vec![
          df::DynVar { param: 0, field: Some(0), erase: false }
        ],
        body: df::DynTerm::Dup {
          expr: Box::new(df::DynTerm::Var { bidx: 0 }),
          body: Box::new(df::DynTerm::Op2 {
            oper: 0,
            val0: Box::new(df::DynTerm::Cal {
              func: 0,
              args: vec![df::DynTerm::Var { bidx: 1 }]
            }),
            val1: Box::new(df::DynTerm::Cal {
              func: 0,
              args: vec![df::DynTerm::Var { bidx: 2 }]
            })
          })
        },
        free: vec![(0, 1)]
      }
    ]
  };

  let stricts = dynfun.redex.clone();

  let rewriter: rt::Rewriter = Box::new(move |mem, host, term| {

    // Gets the left-hand side arguments (ex: `(Succ a)` and `b`)
    //let mut args = Vec::new();
    //for i in 0 .. dynfun.redex.len() {
      //args.push(rt::ask_arg(mem, term, i as u64));
    //}

    // For each argument, if it is redexand a PAR, apply the cal_par rule
    for i in 0 .. dynfun.redex.len() {
      let i = i as u64;
      if dynfun.redex[i as usize] && rt::get_tag(rt::ask_arg(mem,term,i)) == rt::PAR {
        rt::cal_par(mem, host, term, rt::ask_arg(mem,term,i), i as u64);
        return true;
      }
    }

    // For each rule condition vector
    for dynrule in &dynfun.rules {
      // Check if the rule matches
      let mut matched = true;

      // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
      //println!(">> testing conditions... total: {} conds", dynrule.cond.len());
      for (i, cond) in dynrule.cond.iter().enumerate() {
        let i = i as u64;
        match rt::get_tag(*cond) {
          rt::U32 => {
            //println!(">>> cond demands U32 {} at {}", rt::get_val(*cond), i);
            let same_tag = rt::get_tag(rt::ask_arg(mem,term,i)) == rt::U32;
            let same_val = rt::get_val(rt::ask_arg(mem,term,i)) == rt::get_val(*cond);
            matched = matched && same_tag && same_val;
          }
          rt::CTR => {
            //println!(">>> cond demands CTR {} at {}", rt::get_ext(*cond), i);
            let same_tag = rt::get_tag(rt::ask_arg(mem,term,i)) == rt::CTR;
            let same_ext = rt::get_ext(rt::ask_arg(mem,term,i)) == rt::get_ext(*cond);
            matched = matched && same_tag && same_ext;
          }
          _ => {}
        }
      }

      //println!(">> matched? {}", matched);

      // If all conditions are satisfied, the rule matched, so we must apply it
      if matched {
        // Increments the gas count
        rt::inc_cost(mem);
        
        // Gets all the left-hand side vars (ex: `a` and `b`).
        let mut vars = dynrule.vars.iter().map(|df::DynVar {param, field, erase}| 
          match field {
            Some(i) => rt::ask_arg(mem, rt::ask_arg(mem,term,*param), *i),
            None => rt::ask_arg(mem,term,*param),
          }).collect();

        // FIXME: `dups` must be global to properly color the fan nodes, but Rust complains about
        // mutating borrowed variables. Until this is fixed, the language will be very restrict.
        let mut dups = 0;

        // Builds the right-hand side term (ex: `(Succ (Add a b))`)
        //println!("building {:?}", &dynrule.body);
        let done = df::build_dynterm(mem, &dynrule.body, &mut vars, &mut dups);

        // Links the host location to it
        rt::link(mem, host, done);

        // Clears the matched ctrs (the `(Succ ...)` and the `(Add ...)` ctrs)
        rt::clear(mem, rt::get_loc(term, 0), dynfun.redex.len() as u64);
        for (i, arity) in &dynrule.free {
          let i = *i as u64;
          rt::clear(mem, rt::get_loc(rt::ask_arg(mem,term,i), 0), *arity);
        }

        // Collects unused variables (none in this example)
        for (i, df::DynVar {param, field, erase}) in dynrule.vars.iter().enumerate() {
          if *erase {
            rt::collect(mem, vars[i]);
          }
        }

        return true;
      }
    }
    false
  });

  rt::Function { stricts, rewriter }
}

