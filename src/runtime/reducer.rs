pub use crate::runtime::{*};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crossbeam::utils::{Backoff};

#[inline(never)]
pub fn reduce_body(
  heap: &Heap,
  prog: &Program,
  tids: &[usize],
  root: u64,
  stop: &AtomicBool,
  tid: usize,
) {
  // Visit stacks
  let redex = &heap.rbag;
  let visit = &heap.vstk[tid];
  let mut delay = vec![];

  // Backoff
  let backoff = &Backoff::new();

  // State variables
  let mut work = if tid == tids[0] { true } else { false };
  let mut init = if tid == tids[0] { true } else { false };
  let mut cont = if tid == tids[0] { REDEX_CONT_RET } else { 0 };
  let mut host = if tid == tids[0] { root } else { 0 };

  //let mut tick = 0;
  'main: loop {
    //tick = tick + 1;
    //let debug = tid == 9 && heap.lvar[tid].cost.load(Ordering::Relaxed) > 140000000 && heap.lvar[tid].cost.load(Ordering::Relaxed) % 10000 == 0;
    //if tick % 1000000 == 0 { println!("[{}] tick={} cost={}", tid, tick / 1000000, heap.lvar[tid].cost.load(Ordering::Relaxed)); }
    //println!("[{}] reduce\n{}\n", tid, show_term(heap, prog, load_ptr(heap, root), load_ptr(heap, host)));
    //println!("[{}] loop {:?}", tid, &heap.node[0 .. 256]);
    
    //if debug {
      //println!("[{}] loop tick={} cost={} work={} init={} cont={} host={} | {}", tid, tick / 100000000, heap.lvar[tid].cost.load(Ordering::Relaxed), work, init, cont, host, show_ptr(load_ptr(heap, host)));
    //}
    
    if work {
      //println!("[{}] reduce {}", tid, crate::runtime::debug::show_ptr(load_ptr(heap, host)));
      //for (i,name) in prog.nams.data.iter().enumerate() {
        //println!("- {} {:?}", i, name);
      //}
      if init {
        let term = load_ptr(heap, host);
        //println!("[{}] work={} init={} cont={} host={}", tid, work, init, cont, host);
        match get_tag(term) {
          APP => {
            let goup = redex.insert(tid, new_redex(host, cont, 1));
            work = true;
            init = true;
            cont = goup;
            host = get_loc(term, 0);
            continue 'main;
          }
          DP0 | DP1 => {
            match acquire_lock(heap, tid, term) {
              Err(locker_tid) => {
                delay.push(new_visit(host, cont));
                work = false;
                init = true;
                continue 'main;
              }
              Ok(_) => {
                // If the term changed, release lock and try again
                if term != load_ptr(heap, host) {
                  release_lock(heap, tid, term);
                  continue 'main;
                }
                let goup = redex.insert(tid, new_redex(host, cont, 1));
                work = true;
                init = true;
                cont = goup;
                host = get_loc(term, 2);
                continue 'main;
              }
            }
          }
          OP2 => {
            let goup = redex.insert(tid, new_redex(host, cont, 2));
            visit.push(new_visit(get_loc(term, 1), goup));
            work = true;
            init = true;
            cont = goup;
            host = get_loc(term, 0);
            continue 'main;
          }
          FUN => {
            let fid = get_ext(term);
            match &prog.funs.get(&fid) {
              Some(Function::Interpreted { arity: fn_arity, visit: fn_visit, apply: fn_apply }) => {
                let len = fn_visit.strict_idx.len() as u64;
                if len == 0 {
                  work = true;
                  init = false;
                  continue 'main;
                } else {
                  let goup = redex.insert(tid, new_redex(host, cont, fn_visit.strict_idx.len() as u64));
                  for (i, arg_idx) in fn_visit.strict_idx.iter().enumerate() {
                    if i < fn_visit.strict_idx.len() - 1 {
                      visit.push(new_visit(get_loc(term, *arg_idx), goup));
                    } else {
                      work = true;
                      init = true;
                      cont = goup;
                      host = get_loc(term, *arg_idx);
                      continue 'main;
                    }
                  }
                }
              }
              Some(Function::Compiled { arity: fn_arity, visit: fn_visit, apply: fn_apply }) => {
                let host = &mut host;
                let work = &mut work;
                let init = &mut init;
                let cont = &mut cont;
                fn_visit(ReduceCtx { heap, prog, tid, host, term, visit, redex, work, init, cont });
                continue 'main;
              }
              None => {}
            }
          }
          _ => {}
        }
        work = true;
        init = false;
        continue 'main;
      } else {
        let term = load_ptr(heap, host);
        //println!("[{}] reduce {} | {}", tid, host, show_ptr(term));
        
        // Apply rewrite rules
        match get_tag(term) {
          APP => {
            let arg0 = load_arg(heap, term, 0);
            if get_tag(arg0) == LAM {
              app_lam::apply(heap, &prog.arit, tid, host, term, arg0);
              work = true;
              init = true;
              continue 'main;
            }
            if get_tag(arg0) == SUP {
              app_sup::apply(heap, &prog.arit, tid, host, term, arg0);
            }
          }
          DP0 | DP1 => {
            let arg0 = load_arg(heap, term, 2);
            let tcol = get_ext(term);
            //println!("[{}] dups {}", lvar.tid, get_loc(term, 0));
            if get_tag(arg0) == LAM {
              dup_lam::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
              release_lock(heap, tid, term);
              work = true;
              init = true;
              continue 'main;
            } else if get_tag(arg0) == SUP {
              //println!("dup-sup {}", tcol == get_ext(arg0));
              if tcol == get_ext(arg0) {
                dup_dup::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
                release_lock(heap, tid, term);
                work = true;
                init = true;
                continue 'main;
              } else {
                dup_sup::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
                release_lock(heap, tid, term);
                work = true;
                init = true;
                continue 'main;
              }
            } else if get_tag(arg0) == NUM {
              dup_num::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
              release_lock(heap, tid, term);
              work = true;
              init = true;
              continue 'main;
            } else if get_tag(arg0) == CTR {
              dup_ctr::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
              release_lock(heap, tid, term);
              work = true;
              init = true;
              continue 'main;
            } else if get_tag(arg0) == ERA {
              dup_era::apply(heap, &prog.arit, tid, host, term, arg0, tcol);
              release_lock(heap, tid, term);
              work = true;
              init = true;
              continue 'main;
            } else {
              release_lock(heap, tid, term);
            }
          }
          OP2 => {
            let arg0 = load_arg(heap, term, 0);
            let arg1 = load_arg(heap, term, 1);
            if get_tag(arg0) == NUM && get_tag(arg1) == NUM {
              op2_num::apply(heap, &prog.arit, tid, host, term, arg0, arg1);
            } else if get_tag(arg0) == SUP {
              op2_sup_0::apply(heap, &prog.arit, tid, host, term, arg0, arg1);
            } else if get_tag(arg1) == SUP {
              op2_sup_1::apply(heap, &prog.arit, tid, host, term, arg0, arg1);
            }
          }
          FUN => {
            let fid = get_ext(term);
            match &prog.funs.get(&fid) {
              Some(Function::Interpreted { arity: fn_arity, visit: fn_visit, apply: fn_apply }) => {
                if fun_ctr::apply(heap, prog, tid, host, term, fid, *fn_arity, fn_visit, fn_apply) {
                  work = true;
                  init = true;
                  continue 'main;
                }
              }
              Some(Function::Compiled { arity: fn_arity, visit: fn_visit, apply: fn_apply }) => {
                let host = &mut host;
                let work = &mut work;
                let init = &mut init;
                let cont = &mut cont;
                if fn_apply(ReduceCtx { heap, prog, tid, host, term, visit, redex, work, init, cont }) {
                  continue 'main;
                }
              }
              None => {}
            }
          }
          _ => {}
        }

        // If root is on WHNF, halt
        if cont == REDEX_CONT_RET {
          stop.store(true, Ordering::Relaxed);
          break;
        }

        // Otherwise, try reducing the parent redex
        if let Some((new_host, new_cont)) = redex.complete(cont) {
          work = true;
          init = false;
          host = new_host;
          cont = new_cont;
          continue 'main;
        }

        // Otherwise, visit next pointer
        work = false;
        init = true;
        continue 'main;
      }
    } else {
      if init {
        // If available, visit a new location
        if let Some((new_host, new_cont)) = visit.pop() {
          work = true;
          init = true;
          host = new_host;
          cont = new_cont;
          continue 'main;
        }
        // If available, visit a delayed location
        if delay.len() > 0 {
          for next in delay.drain(0..).rev() {
            visit.push(next);
          }
          work = false;
          init = true;
          continue 'main;
        }
        // Otherwise, we have nothing to do
        work = false;
        init = false;
        continue 'main;
      } else {
        if stop.load(Ordering::Relaxed) {
          break;
        } else {
          for victim_tid in tids {
            if *victim_tid != tid {
              if let Some((new_host, new_cont)) = heap.vstk[*victim_tid].steal() {
                //println!("[{}] stole {} {} from {}", tid, new_host, new_cont, victim_tid);
                work = true;
                init = true;
                host = new_host;
                cont = new_cont;
                continue 'main;
              }
            }
          }
          backoff.snooze();
          continue 'main;
        }
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
        reduce_body(heap, prog, tids, root, stop, *tid);
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
