pub use crate::runtime::{*};
use crossbeam::utils::{Backoff};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering};

pub struct ReduceCtx<'a> {
  pub heap  : &'a Heap,
  pub prog  : &'a Program,
  pub tid   : usize,
  pub hold  : bool,
  pub term  : Ptr,
  pub visit : &'a VisitQueue,
  pub redex : &'a RedexBag,
  pub cont  : &'a mut u64,
  pub host  : &'a mut u64,
}

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

pub fn is_whnf(term: Ptr) -> bool {
  match get_tag(term) {
    ERA => true,
    LAM => true,
    SUP => true,
    CTR => true,
    U60 => true,
    F60 => true,
    _   => false,
  }
}

pub fn reduce(heap: &Heap, prog: &Program, tids: &[usize], root: u64, full: bool, debug: bool) -> Ptr {
  // Halting flag
  let stop = &AtomicUsize::new(1);
  let barr = &Barrier::new(tids.len());
  let locs = &tids.iter().map(|x| AtomicU64::new(u64::MAX)).collect::<Vec<AtomicU64>>();

  // Spawn a thread for each worker
  std::thread::scope(|s| {
    for tid in tids {
      s.spawn(move || {
        reducer(heap, prog, tids, stop, barr, locs, root, *tid, full, debug);
        //println!("[{}] done", tid);
      });
    }
  });

  // Return whnf term ptr
  return load_ptr(heap, root);
}

pub fn reducer(
  heap: &Heap,
  prog: &Program,
  tids: &[usize],
  stop: &AtomicUsize,
  barr: &Barrier,
  locs: &[AtomicU64],
  root: u64,
  tid: usize,
  full: bool,
  debug: bool,
) {

  // State Stacks
  let redex = &heap.rbag;
  let visit = &heap.vstk[tid];
  let bkoff = &Backoff::new();
  let hold  = tids.len() <= 1;
  let seen  = &mut HashSet::new();

  // State Vars
  let (mut cont, mut host) = if tid == tids[0] {
    (REDEX_CONT_RET, root)
  } else {
    (0, u64::MAX)
  };

  // Debug Printer
  let print = |tid: usize, host: u64| {
    barr.wait(stop);
    locs[tid].store(host, Ordering::SeqCst);
    barr.wait(stop);
    if tid == tids[0] {
      println!("{}\n----------------", show_at(heap, prog, root, locs));
    }
    barr.wait(stop);
  };

  // State Machine
  'main: loop {
    'init: {
      if host == u64::MAX {
        break 'init;
      }
      'work: loop {
        'visit: loop {
          let term = load_ptr(heap, host);
          if debug {
            print(tid, host);
          }
          match get_tag(term) {
            APP => {
              if app::visit(ReduceCtx { heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }) {
                continue 'visit;
              } else {
                break 'work;
              }
            }
            DP0 | DP1 => {
              match acquire_lock(heap, tid, term) {
                Err(locker_tid) => {
                  continue 'work;
                }
                Ok(_) => {
                  // If the term changed, release lock and try again
                  if term != load_ptr(heap, host) {
                    release_lock(heap, tid, term);
                    continue 'visit;
                  } else {
                    if dup::visit(ReduceCtx { heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }) {
                      continue 'visit;
                    } else {
                      break 'work;
                    }
                  }
                }
              }
            }
            OP2 => {
              if op2::visit(ReduceCtx { heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }) {
                continue 'visit;
              } else {
                break 'work;
              }
            }
            FUN | CTR => {
              let fid = get_ext(term);
//[[CODEGEN:FAST-VISIT]]//
              match &prog.funs.get(&fid) {
                Some(Function::Interpreted { smap: fn_smap, visit: fn_visit, apply: fn_apply }) => {
                  if fun::visit(ReduceCtx { heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }, &fn_visit.strict_idx) {
                    continue 'visit;
                  } else {
                    break 'visit;
                  }
                }
                Some(Function::Compiled { smap: fn_smap, visit: fn_visit, apply: fn_apply }) => {
                  if fn_visit(ReduceCtx { heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }) {
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
            let term = load_ptr(heap, host);
            if debug {
              print(tid, host);
            }
            // Apply rewrite rules
            match get_tag(term) {
              APP => {
                if app::apply(ReduceCtx { heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }) {
                  continue 'work;
                } else {
                  break 'apply;
                }
              }
              DP0 | DP1 => {
                if dup::apply(ReduceCtx { heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }) {
                  release_lock(heap, tid, term);
                  continue 'work;
                } else {
                  release_lock(heap, tid, term);
                  break 'apply;
                }
              }
              OP2 => {
                if op2::apply(ReduceCtx { heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }) {
                  continue 'work;
                } else {
                  break 'apply;
                }
              }
              FUN | CTR => {
                let fid = get_ext(term);
//[[CODEGEN:FAST-APPLY]]//
                match &prog.funs.get(&fid) {
                  Some(Function::Interpreted { smap: fn_smap, visit: fn_visit, apply: fn_apply }) => {
                    if fun::apply(ReduceCtx { heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }, fid, fn_visit, fn_apply) {
                      continue 'work;
                    } else {
                      break 'apply;
                    }
                  }
                  Some(Function::Compiled { smap: fn_smap, visit: fn_visit, apply: fn_apply }) => {
                    if fn_apply(ReduceCtx { heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }) {
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
          // If root is on WHNF, halt
          if cont == REDEX_CONT_RET {
            //println!("done {}", show_at(heap, prog, host, &[]));
            stop.fetch_sub(1, Ordering::Relaxed);
            if full && !seen.contains(&host) {
              seen.insert(host);
              let term = load_ptr(heap, host);
              match get_tag(term) {
                LAM => {
                  stop.fetch_add(1, Ordering::Relaxed);
                  visit.push(new_visit(get_loc(term, 1), hold, cont));
                }
                APP => {
                  stop.fetch_add(2, Ordering::Relaxed);
                  visit.push(new_visit(get_loc(term, 0), hold, cont));
                  visit.push(new_visit(get_loc(term, 1), hold, cont));
                }
                SUP => {
                  stop.fetch_add(2, Ordering::Relaxed);
                  visit.push(new_visit(get_loc(term, 0), hold, cont));
                  visit.push(new_visit(get_loc(term, 1), hold, cont));
                }
                DP0 => {
                  stop.fetch_add(1, Ordering::Relaxed);
                  visit.push(new_visit(get_loc(term, 2), hold, cont));
                }
                DP1 => {
                  stop.fetch_add(1, Ordering::Relaxed);
                  visit.push(new_visit(get_loc(term, 2), hold, cont));
                }
                CTR | FUN => {
                  let arit = arity_of(&prog.aris, term);
                  if arit > 0 {
                    stop.fetch_add(arit as usize, Ordering::Relaxed);
                    for i in 0 .. arit {
                      visit.push(new_visit(get_loc(term, i), hold, cont));
                    }
                  }
                }
                _ => {}
              }
            }
            break 'work;
          }
          // Otherwise, try reducing the parent redex
          if let Some((new_cont, new_host)) = redex.complete(cont) {
            cont = new_cont;
            host = new_host;
            continue 'call;
          }
          // Otherwise, visit next pointer
          break 'work;
        }
      }
      'blink: loop {
        // If available, visit a new location
        if let Some((new_cont, new_host)) = visit.pop() {
          cont = new_cont;
          host = new_host;
          continue 'main;
        }
        // Otherwise, we have nothing to do
        else {
          break 'blink;
        }
      }
    }
    'steal: loop {
      if debug {
        //println!("[{}] steal delay={}", tid, delay.len());
        print(tid, u64::MAX);
      }
      //println!("[{}] steal", tid);
      if stop.load(Ordering::Relaxed) == 0 {
        //println!("[{}] stop", tid);
        break 'main;
      } else {
        for victim_tid in tids {
          if *victim_tid != tid {
            if let Some((new_cont, new_host)) = heap.vstk[*victim_tid].steal() {
              cont = new_cont;
              host = new_host;
              //println!("stolen");
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

pub fn normalize(heap: &Heap, prog: &Program, tids: &[usize], host: u64, debug: bool) -> Ptr {
  let mut cost = get_cost(heap);
  loop {
    reduce(heap, prog, tids, host, true, debug);
    let new_cost = get_cost(heap);
    if new_cost != cost {
      cost = new_cost;
    } else {
      break;
    }
  }
  load_ptr(heap, host)
}

//pub fn normal(heap: &Heap, prog: &Program, tids: &[usize], host: u64, seen: &mut im::HashSet<u64>, debug: bool) -> Ptr {
  //let term = load_ptr(heap, host);
  //if seen.contains(&host) {
    //term
  //} else {
    ////let term = reduce2(heap, lvars, prog, host);
    //let term = reduce(heap, prog, tids, host, debug);
    //seen.insert(host);
    //let mut rec_locs = vec![];
    //match get_tag(term) {
      //LAM => {
        //rec_locs.push(get_loc(term, 1));
      //}
      //APP => {
        //rec_locs.push(get_loc(term, 0));
        //rec_locs.push(get_loc(term, 1));
      //}
      //SUP => {
        //rec_locs.push(get_loc(term, 0));
        //rec_locs.push(get_loc(term, 1));
      //}
      //DP0 => {
        //rec_locs.push(get_loc(term, 2));
      //}
      //DP1 => {
        //rec_locs.push(get_loc(term, 2));
      //}
      //CTR | FUN => {
        //let arity = arity_of(&prog.aris, term);
        //for i in 0 .. arity {
          //rec_locs.push(get_loc(term, i));
        //}
      //}
      //_ => {}
    //}
    //let rec_len = rec_locs.len(); // locations where we must recurse
    //let thd_len = tids.len(); // number of available threads
    //let rec_loc = &rec_locs;
    ////println!("~ rec_len={} thd_len={} {}", rec_len, thd_len, show_term(heap, prog, ask_lnk(heap,host), host));
    //if rec_len > 0 {
      //std::thread::scope(|s| {
        //// If there are more threads than rec_locs, splits threads for each rec_loc
        //if thd_len >= rec_len {
          ////panic!("b");
          //let spt_len = thd_len / rec_len;
          //let mut tids = tids;
          //for (rec_num, rec_loc) in rec_loc.iter().enumerate() {
            //let (rec_tids, new_tids) = tids.split_at(if rec_num == rec_len - 1 { tids.len() } else { spt_len });
            ////println!("~ rec_loc {} gets {} threads", rec_loc, rec_lvars.len());
            ////let new_loc;
            ////if thd_len == rec_len {
              ////new_loc = alloc(heap, rec_tids[0], 1);
              ////move_ptr(heap, *rec_loc, new_loc);
            ////} else {
              ////new_loc = *rec_loc;
            ////}
            ////let new_loc = *rec_loc;
            //let mut seen = seen.clone();
            //s.spawn(move || {
              //let ptr = normal(heap, prog, rec_tids, *rec_loc, &mut seen, debug);
              ////if thd_len == rec_len {
                ////move_ptr(heap, new_loc, *rec_loc);
              ////}
              //link(heap, *rec_loc, ptr);
            //});
            //tids = new_tids;
          //}
        //// Otherwise, splits rec_locs for each thread
        //} else {
          ////panic!("c");
          //for (thd_num, tid) in tids.iter().enumerate() {
            //let min_idx = thd_num * rec_len / thd_len;
            //let max_idx = if thd_num < thd_len - 1 { (thd_num + 1) * rec_len / thd_len } else { rec_len };
            ////println!("~ thread {} gets rec_locs {} to {}", thd_num, min_idx, max_idx);
            //let mut seen = seen.clone();
            //s.spawn(move || {
              //for idx in min_idx .. max_idx {
                //let loc = rec_loc[idx];
                //let lnk = normal(heap, prog, std::slice::from_ref(tid), loc, &mut seen, debug);
                //link(heap, loc, lnk);
              //}
            //});
          //}
        //}
      //});
    //}
    //term
  //}
//}

//pub fn normalize(heap: &Heap, prog: &Program, tids: &[usize], host: u64, debug: bool) -> Ptr {
  //let mut cost = get_cost(heap);
  //loop {
    //normal(heap, prog, tids, host, &mut im::HashSet::new(), debug);
    //let new_cost = get_cost(heap);
    //if new_cost != cost {
      //cost = new_cost;
    //} else {
      //break;
    //}
  //}
  //load_ptr(heap, host)
//}
