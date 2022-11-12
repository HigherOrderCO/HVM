pub use crate::runtime::{*};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crossbeam::utils::{Backoff};

// HVM's reducer is a finite stack machine with 4 possible states:
// - visit: visits a node and add its children to the visit stack ~> visit, apply, blink
// - apply: reduces a node, applying a rewrite rule               ~> visit, apply, blink, halt
// - blink: pops the visit stack and enters visit mode            ~> visit, blink, steal
// - steal: attempt to steal work from the global pool            ~> visit, steal, halt
// Since Rust doesn't have `goto`, the loop structure below is used.
// It allows performing any allowed state transition with a jump.
//   main {
//     work {
//       visit { ... }
//       apply { ... }
//       complete
//     }
//     blink { ... }
//     steal { ... }
//   }

// funcoes podem retornar 3 coisas:
// - favor ir para o modo visit, pois gerei filhos
// - favor ir para o modo apply, pois não gerei filhos
// - favor ir para o modo climb, pois estou normalizado

// TODO: rebote: quando volta do apply, tentar um outro apply diretamente

// A function must return `true` if it must visit. Optimize by returning `false` on ctors.
// A visit should attempt applying before? <- maybe not needed if the extra load is skipped

pub fn is_whnf(term: Ptr) -> bool {
  match get_tag(term) {
    VAR => true,
    ERA => true,
    LAM => true,
    SUP => true,
    CTR => true,
    NUM => true,
    _   => false,
  }
}

#[inline(never)]
pub fn reducer(heap: &Heap, prog: &Program, tids: &[usize], stop: &AtomicBool, root: u64, tid: usize) {

  // State Stacks
  let redex = &heap.rbag;
  let visit = &heap.vstk[tid];
  let delay = &mut vec![];
  let bkoff = &Backoff::new();

  // State Vars
  let (mut cont, mut host) = if tid == tids[0] {
    (REDEX_CONT_RET, root)
  } else {
    (0, u64::MAX)
  };

  //let mut tick = 0;

  // State Machine
  'main: loop {
    'init: {
      if host == u64::MAX {
        break 'init;
      }
      //if cont == REDEX_CONT_RET { println!("[{}] ue", tid); }
      //tick = tick + 1; if tick > 20 { std::process::exit(0); };
      //println!("main {} {}", show_ptr(load_ptr(heap, host)), show_term(heap, prog, load_ptr(heap, host), host));
      'work: loop {
        //println!("work {} {}", show_ptr(load_ptr(heap, host)), show_term(heap, prog, load_ptr(heap, host), host));
        'visit: loop {
          let term = load_ptr(heap, host);
          //println!("visit {} {}", show_ptr(load_ptr(heap, host)), show_term(heap, prog, load_ptr(heap, host), host));
          match get_tag(term) {
            APP => {
              let goup = redex.insert(tid, new_redex(host, cont, 1));
              cont = goup;
              host = get_loc(term, 0);
              continue 'visit;
            }
            DP0 | DP1 => {
              match acquire_lock(heap, tid, term) {
                Err(locker_tid) => {
                  delay.push(new_visit(host, cont));
                  break 'work;
                }
                Ok(_) => {
                  // If the term changed, release lock and try again
                  if term != load_ptr(heap, host) {
                    release_lock(heap, tid, term);
                    continue 'visit;
                  } else {
                    let goup = redex.insert(tid, new_redex(host, cont, 1));
                    cont = goup;
                    host = get_loc(term, 2);
                    continue 'visit;
                  }
                }
              }
            }
            OP2 => {
              let goup = redex.insert(tid, new_redex(host, cont, 2));
              visit.push(new_visit(get_loc(term, 1), goup));
              cont = goup;
              host = get_loc(term, 0);
              continue 'visit;
            }
            FUN => {
              let fid = get_ext(term);
//[[CODEGEN:FAST-VISIT]]//
              match &prog.funs.get(&fid) {
                Some(Function::Interpreted { arity: fn_arity, visit: fn_visit, apply: fn_apply }) => {
                  if fun_ctr::visit(ReduceCtx { heap, prog, tid, term, visit, redex, cont: &mut cont, host: &mut host }, &fn_visit.strict_idx) {
                    continue 'visit;
                  } else {
                    break 'visit;
                  }
                }
                Some(Function::Compiled { arity: fn_arity, visit: fn_visit, apply: fn_apply }) => {
                  if fn_visit(ReduceCtx { heap, prog, tid, term, visit, redex, cont: &mut cont, host: &mut host }) {
                    continue 'visit;
                  } else {
                    break 'visit;
                  }
                }
                None => {
                  break 'visit;
                }
              }
            }
            _ => {
              break 'visit;
            }
          }
        }
        'call: loop {
          'apply: loop {
            //println!("apply {} {}", show_ptr(load_ptr(heap, host)), show_term(heap, prog, load_ptr(heap, host), host));
            let term = load_ptr(heap, host);
            // Apply rewrite rules
            match get_tag(term) {
              APP => {
                let arg0 = load_arg(heap, term, 0);
                if get_tag(arg0) == LAM {
                  app_lam::apply(heap, &prog.arit, tid, host, term, arg0);
                  continue 'work;
                }
                if get_tag(arg0) == SUP {
                  app_sup::apply(heap, &prog.arit, tid, host, term, arg0);
                  break 'apply;
                }
              }
              DP0 | DP1 => {
                let arg0 = load_arg(heap, term, 2);
                let tcol = get_ext(term);
                //println!("[{}] dups {}", lvar.tid, get_loc(term, 0));
                if get_tag(arg0) == LAM {
                  dup_lam::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
                  release_lock(heap, tid, term);
                  continue 'work;
                } else if get_tag(arg0) == SUP {
                  //println!("dup-sup {}", tcol == get_ext(arg0));
                  if tcol == get_ext(arg0) {
                    dup_dup::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
                    release_lock(heap, tid, term);
                    continue 'work;
                  } else {
                    dup_sup::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
                    release_lock(heap, tid, term);
                    continue 'work;
                  }
                } else if get_tag(arg0) == NUM {
                  dup_num::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
                  release_lock(heap, tid, term);
                  continue 'work;
                } else if get_tag(arg0) == CTR {
                  dup_ctr::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
                  release_lock(heap, tid, term);
                  continue 'work;
                } else if get_tag(arg0) == ERA {
                  dup_era::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
                  release_lock(heap, tid, term);
                  continue 'work;
                } else {
                  release_lock(heap, tid, term);
                  break 'apply;
                }
              }
              OP2 => {
                let arg0 = load_arg(heap, term, 0);
                let arg1 = load_arg(heap, term, 1);
                if get_tag(arg0) == NUM && get_tag(arg1) == NUM {
                  op2_num::apply(heap, &prog.arit, tid, host, term, arg0, arg1);
                  break 'apply;
                } else if get_tag(arg0) == SUP {
                  op2_sup_0::apply(heap, &prog.arit, tid, host, term, arg0, arg1);
                  break 'apply;
                } else if get_tag(arg1) == SUP {
                  op2_sup_1::apply(heap, &prog.arit, tid, host, term, arg0, arg1);
                  break 'apply;
                }
              }
              FUN => {
                let fid = get_ext(term);
//[[CODEGEN:FAST-APPLY]]//
                match &prog.funs.get(&fid) {
                  Some(Function::Interpreted { arity: fn_arity, visit: fn_visit, apply: fn_apply }) => {
                    if fun_ctr::apply(heap, prog, tid, host, term, fid, *fn_arity, fn_visit, fn_apply) {
                      continue 'work;
                    } else {
                      break 'apply;
                    }
                  }
                  Some(Function::Compiled { arity: fn_arity, visit: fn_visit, apply: fn_apply }) => {
                    if fn_apply(ReduceCtx { heap, prog, tid, term, visit, redex, cont: &mut cont, host: &mut host }) {
                      continue 'work;
                    } else {
                      break 'apply;
                    }
                  }
                  None => {
                    break 'apply;
                  }
                }
              }
              _ => {
                break 'apply;
              }
            }
          }
          //println!("climb {} {}", show_ptr(load_ptr(heap, host)), show_term(heap, prog, load_ptr(heap, host), host));
          // If root is on WHNF, halt
          if cont == REDEX_CONT_RET {
            //println!("[{}] stop {} {}", tid, cont, REDEX_CONT_RET);
            stop.store(true, Ordering::Relaxed);
            break 'main;
          }
          // Otherwise, try reducing the parent redex
          else if let Some((new_cont, new_host)) = redex.complete(cont) {
            cont = new_cont;
            host = new_host;
            continue 'call;
          }
          // Otherwise, visit next pointer
          else {
            break 'work;
          }
        }
      }
      'blink: loop {
        //println!("blink");
        // If available, visit a new location
        if let Some((new_cont, new_host)) = visit.pop() {
          cont = new_cont;
          host = new_host;
          continue 'main;
        }
        // If available, visit a delayed location
        else if delay.len() > 0 {
          for next in delay.drain(0..).rev() {
            visit.push(next);
          }
          continue 'blink;
        }
        // Otherwise, we have nothing to do
        else {
          break 'blink;
        }
      }
    }
    'steal: loop {
      //println!("[{}] steal", tid);
      if stop.load(Ordering::Relaxed) {
        //println!("[{}] halt", tid);
        break 'main;
      } else {
        for victim_tid in tids {
          if *victim_tid != tid {
            if let Some((new_cont, new_host)) = heap.vstk[*victim_tid].steal() {
              //println!("[{}] stole {} {} from {}", tid, new_host, new_cont, victim_tid);
              cont = new_cont;
              host = new_host;
              continue 'main;
            }
          }
        }
        bkoff.snooze();
        continue 'steal;
      }
    }
  }
}

pub fn reduce(heap: &Heap, prog: &Program, tids: &[usize], root: u64) -> Ptr {
  // Halting flag
  let stop = &AtomicBool::new(false);

  // Spawn a thread for each worker
  std::thread::scope(|s| {
    for tid in tids {
      s.spawn(move || {
        reducer(heap, prog, tids, stop, root, *tid);
      });
    }
  });

  return load_ptr(heap, root);
}

pub fn normal(heap: &Heap, prog: &Program, tids: &[usize], host: u64, visited: &Box<[AtomicU64]>) -> Ptr {
  pub fn set_visited(visited: &Box<[AtomicU64]>, bit: u64) {
    let val = &visited[bit as usize >> 6];
    val.store(val.load(Ordering::Relaxed) | (1 << (bit & 0x3f)), Ordering::Relaxed);
  }
  pub fn was_visited(visited: &Box<[AtomicU64]>, bit: u64) -> bool {
    let val = &visited[bit as usize >> 6];
    (((val.load(Ordering::Relaxed) >> (bit & 0x3f)) as u8) & 1) == 1
  }
  let term = load_ptr(heap, host);
  if was_visited(visited, host) {
    term
  } else {
    //let term = reduce2(heap, lvars, prog, host);
    let term = reduce(heap, prog, tids, host);
    set_visited(visited, host);
    let mut rec_locs = vec![];
    match get_tag(term) {
      LAM => {
        rec_locs.push(get_loc(term, 1));
      }
      APP => {
        rec_locs.push(get_loc(term, 0));
        rec_locs.push(get_loc(term, 1));
      }
      SUP => {
        rec_locs.push(get_loc(term, 0));
        rec_locs.push(get_loc(term, 1));
      }
      DP0 => {
        rec_locs.push(get_loc(term, 2));
      }
      DP1 => {
        rec_locs.push(get_loc(term, 2));
      }
      CTR | FUN => {
        let arity = arity_of(&prog.arit, term);
        for i in 0 .. arity {
          rec_locs.push(get_loc(term, i));
        }
      }
      _ => {}
    }
    let rec_len = rec_locs.len(); // locations where we must recurse
    let thd_len = tids.len(); // number of available threads
    let rec_loc = &rec_locs;
    //println!("~ rec_len={} thd_len={} {}", rec_len, thd_len, show_term(heap, prog, ask_lnk(heap,host), host));
    if rec_len > 0 {
      std::thread::scope(|s| {
        // If there are more threads than rec_locs, splits threads for each rec_loc
        if thd_len >= rec_len {
          //panic!("b");
          let spt_len = thd_len / rec_len;
          let mut tids = tids;
          for (rec_num, rec_loc) in rec_loc.iter().enumerate() {
            let (rec_tids, new_tids) = tids.split_at(if rec_num == rec_len - 1 { tids.len() } else { spt_len });
            //println!("~ rec_loc {} gets {} threads", rec_loc, rec_lvars.len());
            //let new_loc;
            //if thd_len == rec_len {
              //new_loc = alloc(heap, rec_tids[0], 1);
              //move_ptr(heap, *rec_loc, new_loc);
            //} else {
              //new_loc = *rec_loc;
            //}
            let new_loc = *rec_loc;
            s.spawn(move || {
              let ptr = normal(heap, prog, rec_tids, new_loc, visited);
              //if thd_len == rec_len {
                //move_ptr(heap, new_loc, *rec_loc);
              //}
              link(heap, *rec_loc, ptr);
            });
            tids = new_tids;
          }
        // Otherwise, splits rec_locs for each thread
        } else {
          //panic!("c");
          for (thd_num, tid) in tids.iter().enumerate() {
            let min_idx = thd_num * rec_len / thd_len;
            let max_idx = if thd_num < thd_len - 1 { (thd_num + 1) * rec_len / thd_len } else { rec_len };
            //println!("~ thread {} gets rec_locs {} to {}", thd_num, min_idx, max_idx);
            s.spawn(move || {
              for idx in min_idx .. max_idx {
                let loc = rec_loc[idx];
                let lnk = normal(heap, prog, std::slice::from_ref(tid), loc, visited);
                link(heap, loc, lnk);
              }
            });
          }
        }
      });
    }
    term
  }
}

pub fn normalize(heap: &Heap, prog: &Program, tids: &[usize], host: u64, run_io: bool) -> Ptr {
  let mut cost = get_cost(heap);
  let visited = new_atomic_u64_array(HEAP_SIZE / 64);
  loop {
    let visited = new_atomic_u64_array(HEAP_SIZE / 64);
    normal(&heap, prog, tids, host, &visited);
    let new_cost = get_cost(heap);
    if new_cost != cost {
      cost = new_cost;
    } else {
      break;
    }
  }
  load_ptr(heap, host)
}
