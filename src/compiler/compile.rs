// TODO: optimize apply to return false when it is a ctr
// TODO: optimize apply to realloc same arity nodes

use std::collections::HashMap;
use crate::language as language;
use crate::runtime as runtime;

pub fn build_name(name: &str) -> String {
  // TODO: this can still cause some name collisions.
  // Note: avoiding the use of `$` because it is not an actually valid
  // identifier character in C.
  //let name = name.replace('_', "__");
  let name = name.replace('.', "_").replace('$', "_S_");
  format!("_{}_", name)
}

pub fn build_code(code: &str) -> Result<(String,String), String> {
  let file = language::syntax::read_file(code)?;
  let book = language::rulebook::gen_rulebook(&file);
  runtime::gen_functions(&book);
  Ok(build_rulebook(&book))
}

pub fn build_rulebook(book: &language::rulebook::RuleBook) -> (String, String) {
  // precomp ids
  let mut precomp_ids = String::new();
  for (id, name) in itertools::sorted(book.id_to_name.iter()) {
    //if id >= &runtime::PRECOMP_COUNT {
    line(&mut precomp_ids, 0, &format!("pub const {} : u64 = {};", &build_name(name), id));
    //}
  }

  // precomp els
  let mut precomp_els = String::new();
  for id in itertools::sorted(book.id_to_name.keys()) {
    if id >= &runtime::PRECOMP_COUNT {
      let name = book.id_to_name.get(id).unwrap();
      line(&mut precomp_els, 0, &format!(r#"  Precomp {{"#));
      line(&mut precomp_els, 0, &format!(r#"    id: {},"#, &build_name(&name)));
      line(&mut precomp_els, 0, &format!(r#"    name: "{}","#, &name));
      line(&mut precomp_els, 0, &format!(r#"    smap: &{:?},"#, book.id_to_smap.get(id).unwrap()));
      if *book.ctr_is_fun.get(name).unwrap_or(&false) {
        line(&mut precomp_els, 0, &format!(r#"    funs: Some(PrecompFuns {{"#));
        line(&mut precomp_els, 0, &format!(r#"      visit: {}_visit,"#, &build_name(&name)));
        line(&mut precomp_els, 0, &format!(r#"      apply: {}_apply,"#, &build_name(&name)));
        line(&mut precomp_els, 0, &format!(r#"    }}),"#));
      } else {
        line(&mut precomp_els, 0, &format!(r#"    funs: None,"#));
      }
      line(&mut precomp_els, 0, &format!(r#"  }},"#));
    }
  }

  // precomp fns
  let mut precomp_fns = String::new();
  for id in itertools::sorted(book.id_to_name.keys()) {
    if id >= &runtime::PRECOMP_COUNT {
      let name = book.id_to_name.get(id).unwrap();
      if let Some(rules) = book.rule_group.get(name) {
        let (got_visit, got_apply) = build_function(book, &name, &rules.1);
        line(&mut precomp_fns, 0, &format!("{}", got_visit));
        line(&mut precomp_fns, 0, &format!("{}", got_apply));
      }
    }
  }

  // fast visit
  let mut fast_visit = String::new();
  line(&mut fast_visit, 7, &format!("match fid {{"));
  for id in itertools::sorted(book.id_to_name.keys()) {
    if id >= &runtime::PRECOMP_COUNT {
      let name = book.id_to_name.get(id).unwrap();
      if let Some(rules) = book.rule_group.get(name) {
        line(&mut fast_visit, 8, &format!("{} => {{", &build_name(&name)));
        line(&mut fast_visit, 9, &format!("if {}_visit(ReduceCtx {{ heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }}) {{", &build_name(&name)));
        line(&mut fast_visit, 10, &format!("continue 'visit;"));
        line(&mut fast_visit, 9, &format!("}} else {{"));
        line(&mut fast_visit, 10, &format!("break 'visit;"));
        line(&mut fast_visit, 9, &format!("}}"));
        line(&mut fast_visit, 8, &format!("}}"));
      };
    }
  }
  line(&mut fast_visit, 8, &format!("_ => {{}}"));
  line(&mut fast_visit, 7, &format!("}}"));

  // fast apply
  let mut fast_apply = String::new();
  line(&mut fast_apply, 8, &format!("match fid {{"));
  for id in itertools::sorted(book.id_to_name.keys()) {
    if id >= &runtime::PRECOMP_COUNT {
      let name = book.id_to_name.get(id).unwrap();
      if let Some(rules) = book.rule_group.get(name) {
        line(&mut fast_apply, 9, &format!("{} => {{", &build_name(&name)));
        line(&mut fast_apply, 10, &format!("if {}_apply(ReduceCtx {{ heap, prog, tid, hold, term, visit, redex, cont: &mut cont, host: &mut host }}) {{", &build_name(&name)));
        line(&mut fast_apply, 11, &format!("continue 'work;"));
        line(&mut fast_apply, 10, &format!("}} else {{"));
        line(&mut fast_apply, 11, &format!("break 'apply;"));
        line(&mut fast_apply, 10, &format!("}}"));
        line(&mut fast_apply, 9, &format!("}}"));
      };
    }
  }
  line(&mut fast_apply, 9, &format!("_ => {{}}"));
  line(&mut fast_apply, 8, &format!("}}"));

  // precomp.rs
  let precomp_rs : &str = include_str!("./../runtime/base/precomp.rs");
  let precomp_rs = precomp_rs.replace("//[[CODEGEN:PRECOMP-IDS]]//\n", &precomp_ids);
  let precomp_rs = precomp_rs.replace("//[[CODEGEN:PRECOMP-ELS]]//\n", &precomp_els);
  let precomp_rs = precomp_rs.replace("//[[CODEGEN:PRECOMP-FNS]]//\n", &precomp_fns);

  // reducer.rs
  let reducer_rs : &str = include_str!("./../runtime/base/reducer.rs");
  let reducer_rs = reducer_rs.replace("//[[CODEGEN:FAST-VISIT]]//\n", &fast_visit);
  let reducer_rs = reducer_rs.replace("//[[CODEGEN:FAST-APPLY]]//\n", &fast_apply);

  return (precomp_rs, reducer_rs);
}

pub fn build_function(
  book  : &language::rulebook::RuleBook,
  fname : &str,
  rules : &[language::syntax::Rule],
) -> (String, String) {
  if let runtime::Function::Interpreted {
    smap: fn_smap,
    visit: fn_visit,
    apply: fn_apply,
  } = runtime::build_function(book, fname, rules) {

    // Visit
    // -----

    let mut visit = String::new();
    line(&mut visit, 0, &format!("#[inline(always)]"));
    line(&mut visit, 0, &format!("pub fn {}_visit(ctx: ReduceCtx) -> bool {{", &build_name(fname)));
    if fn_visit.strict_idx.is_empty() {
      line(&mut visit, 1, "return false;");
    } else {
      line(&mut visit, 1, &format!("let mut vlen = 0;"));
      line(&mut visit, 1, &format!("let vbuf = unsafe {{ ctx.heap.vbuf.get_unchecked(ctx.tid) }};"));
      for sidx in &fn_visit.strict_idx {
        line(&mut visit, 1, &format!("if !is_whnf(load_arg(ctx.heap, ctx.term, {})) {{", *sidx));
        line(&mut visit, 2, &format!("unsafe {{ vbuf.get_unchecked(vlen) }}.store(get_loc(ctx.term, {}), Ordering::Relaxed);", *sidx));
        line(&mut visit, 2, &format!("vlen += 1"));
        line(&mut visit, 1, &format!("}}"));
      }
      line(&mut visit, 1, &format!("if vlen == 0 {{"));
      line(&mut visit, 2, &format!("return false;"));
      line(&mut visit, 1, &format!("}} else {{"));
      line(&mut visit, 2, &format!("let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, vlen as u64));"));
      for i in 0 .. fn_visit.strict_idx.len() {
        line(&mut visit, 2, &format!("if {} < vlen - 1 {{", i));
        line(&mut visit, 3, &format!("ctx.visit.push(new_visit(unsafe {{ vbuf.get_unchecked({}).load(Ordering::Relaxed) }}, ctx.hold, goup));", i));
        line(&mut visit, 2, &format!("}}"));
      }
      line(&mut visit, 2, &format!("*ctx.cont = goup;"));
      line(&mut visit, 2, &format!("*ctx.host = unsafe {{ vbuf.get_unchecked(vlen - 1).load(Ordering::Relaxed) }};"));
      line(&mut visit, 2, &format!("return true;"));
      line(&mut visit, 1, &format!("}}"));
    }
    line(&mut visit, 0, &format!("}}"));

    // Apply
    // -----
    
    let mut apply = String::new();

    // Transmute Optimization
    // ----------------------
    // When a function has the shape:
    // (Foo a b c ...) = (Bar a b c ...)
    // It just transmutes the pointer.
    
    'TransmuteOptimization: {
      if fn_apply.rules.len() != 1 {
        break 'TransmuteOptimization;
      }
      let runtime::program::Rule { hoas, cond, vars, core, body, .. } = &fn_apply.rules[0];
      // Checks if it doesn't do any pattern-matching
      for mat in cond {
        if *mat != runtime::Var(0) {
          break 'TransmuteOptimization;
        }
      }
      // Checks if its rhs only allocs one node
      if body.1.len() != 1 {
        break 'TransmuteOptimization;
      }
      // Checks if the function and body arity match
      let cell = &body.1[0];
      if cell.len() != vars.len() {
        break 'TransmuteOptimization;
      }
      // Checks if it returns the same variables in order
      for i in 0 .. cell.len() {
        if let runtime::RuleBodyCell::Var { index } = cell[i] {
          if index != i as u64 {
            break 'TransmuteOptimization;
          }
        } else {
          break 'TransmuteOptimization;
        }
      }
      // Gets the new ptr
      let ptr;
      if let runtime::RuleBodyCell::Ptr { value, targ: 0, slot: 0 } = body.0 {
        ptr = value;
      } else {
        break 'TransmuteOptimization;
      }
      // If all is true, compile as a transmuter
      line(&mut apply, 0, &format!("#[inline(always)]"));
      line(&mut apply, 0, &format!("pub fn {}_apply(ctx: ReduceCtx) -> bool {{", &build_name(fname)));
      line(&mut apply, 1, &format!("let done = Ctr({}, get_loc(ctx.term, 0));", runtime::get_ext(ptr)));
      line(&mut apply, 1, "link(ctx.heap, *ctx.host, done);");
      line(&mut apply, 1, "return false;");
      line(&mut apply, 0, &format!("}}"));
    }

    // Normal Function
    // ---------------
    
    if apply.len() == 0 {
      line(&mut apply, 0, &format!("#[inline(always)]"));
      line(&mut apply, 0, &format!("pub fn {}_apply(ctx: ReduceCtx) -> bool {{", &build_name(fname)));

      // Loads strict arguments
      for i in 0 .. fn_smap.len() {
        line(&mut apply, 1, &format!("let arg{} = load_arg(ctx.heap, ctx.term, {});", i, i));
      }

      // Applies the fun_sup rule to superposed args
      for (i, is_strict) in fn_visit.strict_map.iter().enumerate() {
        if *is_strict {
          line(&mut apply, 1, &format!("if get_tag(arg{}) == SUP {{", i));
          line(&mut apply, 2, &format!("fun::superpose(ctx.heap, &ctx.prog.aris, ctx.tid, *ctx.host, ctx.term, arg{}, {});", i, i));
          line(&mut apply, 1, "}");
        }
      }

      // For each rule condition vector
      for (r, rule) in fn_apply.rules.iter().enumerate() {
        let mut matched: Vec<String> = Vec::new();

        // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
        for (i, cond) in rule.cond.iter().enumerate() {
          let i = i as u64;
          if runtime::get_tag(*cond) == runtime::U60 {
            let same_tag = format!("get_tag(arg{}) == U60", i);
            let same_val = format!("get_num(arg{}) == {}", i, runtime::get_num(*cond));
            matched.push(format!("({} && {})", same_tag, same_val));
          }
          if runtime::get_tag(*cond) == runtime::F60 {
            let same_tag = format!("get_tag(arg{}) == F60", i);
            let same_val = format!("get_num(arg{}) == {}", i, runtime::get_num(*cond));
            matched.push(format!("({} && {})", same_tag, same_val));
          }
          if runtime::get_tag(*cond) == runtime::CTR {
            let some_tag = format!("get_tag(arg{}) == CTR", i);
            let some_ext = format!("get_ext(arg{}) == {}", i, runtime::get_ext(*cond));
            matched.push(format!("({} && {})", some_tag, some_ext));
          }
            // If this is a strict argument, then we're in a default variable
          if runtime::get_tag(*cond) == runtime::VAR && fn_visit.strict_map[i as usize] {

            // This is a Kind2-specific optimization. Check 'HOAS_OPT'.
            if rule.hoas && r != fn_apply.rules.len() - 1 {

              // Matches number literals
              let is_num = format!("({} || {})",
                format!("get_tag(arg{}) == U60", i),
                format!("get_tag(arg{}) == F60", i));

              // Matches constructor labels
              let is_ctr = format!("({} && {})",
                format!("get_tag(arg{}) == CTR", i),
                format!("arity_of(heap, arg{}) == 0u", i));

              // Matches HOAS numbers and constructors
              let is_hoas_ctr_num = format!("({} && {} && {})",
                format!("get_tag(arg{}) == CTR", i),
                format!("get_ext(arg{}) >= HOAS_CT0", i),
                format!("get_ext(arg{}) <= HOAS_F60", i));

              matched.push(format!("({} || {} || {})", is_num, is_ctr, is_hoas_ctr_num));

            // Only match default variables on CTRs, U60s, F60s
            } else {
              let is_ctr = format!("get_tag(arg{}) == CTR", i);
              let is_u60 = format!("get_tag(arg{}) == U60", i);
              let is_f60 = format!("get_tag(arg{}) == F60", i);
              matched.push(format!("({} || {} || {})", is_ctr, is_u60, is_f60));
            }

          }
        }

        let conds = if matched.is_empty() { String::from("true") } else { matched.join(" && ") };
        line(&mut apply, 1, &format!("if {} {{", conds));

        // Increments the gas count
        line(&mut apply, 2, "inc_cost(ctx.heap, ctx.tid);");

        // Builds the free vector
        let mut free : Vec<Option<(String,u64)>> = vec![];
        for (idx, ari) in &rule.free {
          free.push(Some((format!("get_loc(arg{}, 0)", idx), *ari)));
        }
        free.push(Some(("get_loc(ctx.term, 0)".to_string(), fn_visit.strict_map.len() as u64)));

        // Builds the right-hand side term (ex: `(Succ (Add a b))`)
        //let done = build_function_rule_body(&mut apply, 2, &rule.body, &rule.vars);
        let done = build_function_rule_rhs(book, &mut apply, &mut free, 2, &rule.core, &rule.vars);
        line(&mut apply, 2, &format!("let done = {};", done));

        // Links the host location to it
        line(&mut apply, 2, "link(ctx.heap, *ctx.host, done);");

        // Collects unused variables (none in this example)
        for dynvar @ runtime::RuleVar { param: _, field: _, erase } in rule.vars.iter() {
          if *erase {
            line(&mut apply, 2, &format!("collect(ctx.heap, &ctx.prog.aris, ctx.tid, {});", get_var(dynvar)));
          }
        }

        // Clears the matched ctrs (the `(Succ ...)` and the `(Add ...)` ctrs)
        for must_free in &free {
          if let Some((loc, ari)) = must_free {
            line(&mut apply, 2, &format!("free(ctx.heap, ctx.tid, {}, {});", loc, ari));
          }
        }
        
        //for (i, arity) in &rule.free {
          //let i = *i as u64;
          //line(&mut apply, 2, &format!("free(ctx.heap, ctx.tid, get_loc(arg{}, 0), {});", i, arity));
        //}
        //line(&mut apply, 2, &format!("free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), {});", fn_visit.strict_map.len()));
        
        let ret_ptr = match rule.body.0 {
          runtime::RuleBodyCell::Val { value }     => value,
          runtime::RuleBodyCell::Ptr { value, .. } => value,
          runtime::RuleBodyCell::Var { .. }        => runtime::Var(0),
        };
        line(&mut apply, 2, &format!("return {};", if runtime::is_whnf(ret_ptr) { "false" } else { "true" }));
        //line(&mut apply, 2, &format!("return true;"));
        line(&mut apply, 1, "}");
      }
      line(&mut apply, 1, "return false;");

      line(&mut apply, 0, "}");
    }

    (visit, apply)
  } else {
    panic!("Unexpected error.");
  }
}

pub fn build_function_rule_rhs(
  book : &language::rulebook::RuleBook,
  code : &mut String,
  free : &mut Vec<Option<(String,u64)>>,
  tab  : u64,
  term : &runtime::Core,
  rvrs : &[runtime::RuleVar],
) -> String {
  fn alloc_lam(
    code : &mut String,
    tab  : u64,
    free : &mut Vec<Option<(String,u64)>>,
    nams : &mut u64,
    lams : &mut HashMap<u64, String>,
    glob : u64,
  ) -> String {
    if let Some(got) = lams.get(&glob) {
      got.clone()
    } else {
      let name = fresh(nams, "lam");
      line(code, tab, &format!("let {} = {};", name, alloc_node(free, 2)));
      if glob != 0 {
        // FIXME: sanitizer still can't detect if a scopeless lambda doesn't use its bound
        // variable, so we must write an Era() here. When it does, we can remove this line.
        line(code, tab, &format!("link(heap, {} + 0, Era());", name));
        lams.insert(glob, name.clone());
      }
      name
    }
  }
  fn alloc_dup(
    code : &mut String,
    tab  : u64,
    free : &mut Vec<Option<(String,u64)>>,
    nams : &mut u64,
    dups : &mut HashMap<u64, (String,String)>,
    glob : u64,
  ) -> (String, String) {
    if let Some(got) = dups.get(&glob) {
      return got.clone();
    } else {
      let coln = fresh(nams, "col");
      let name = fresh(nams, "dup");
      line(code, tab + 1, &format!("let {} = gen_dup(ctx.heap, ctx.tid);", coln));
      line(code, tab + 1, &format!("let {} = {};", name, alloc_node(free, 3)));
      line(code, tab, &format!("link(ctx.heap, {} + 0, Era());", name)); // FIXME: remove when possible (same as above)
      line(code, tab, &format!("link(ctx.heap, {} + 1, Era());", name)); // FIXME: remove when possible (same as above)
      if glob != 0 {
        dups.insert(glob, (coln.clone(), name.clone()));
      }
      return (coln, name);
    }
  }
  fn alloc_node(
    free: &mut Vec<Option<(String,u64)>>,
    arit: u64,
  ) -> String {
    // This will avoid calls to alloc() by reusing nodes from the left-hand side. Sadly, this seems
    // to decrease HVM's performance in some cases, probably because of added cache misses. Perhaps
    // this should be turned off. I'll decide later.
    for i in 0 .. free.len() {
      if let Some((loc, ari)) = free[i].clone() {
        if ari == arit {
          free[i] = None;
          return format!("{}/*reuse:{}*/", loc.clone(), arit);
        }
      }
    }
    return format!("alloc(ctx.heap, ctx.tid, {})", arit);
  }
  fn build_term(
    book : &language::rulebook::RuleBook,
    code : &mut String,
    tab  : u64,
    free : &mut Vec<Option<(String,u64)>>,
    vars : &mut Vec<String>,
    nams : &mut u64,
    lams : &mut HashMap<u64, String>,
    dups : &mut HashMap<u64, (String,String)>,
    term : &runtime::Core,
  ) -> String {
    const INLINE_NUMBERS: bool = true;

    //println!("compile {:?}", term);
    //println!("- vars: {:?}", vars);
    match term {
      runtime::Core::Var { bidx } => {
        if *bidx < vars.len() as u64 {
          vars[*bidx as usize].clone()
        } else {
          panic!("Unbound variable.");
        }
      }
      runtime::Core::Glo { glob, misc } => {
        match *misc {
          runtime::VAR => {
            return format!("Var({})", alloc_lam(code, tab, free, nams, lams, *glob));
          }
          runtime::DP0 => {
            let (coln, name) = alloc_dup(code, tab, free, nams, dups, *glob);
            return format!("Dp0({}, {})", coln, name);
          }
          runtime::DP1 => {
            let (coln, name) = alloc_dup(code, tab, free, nams, dups, *glob);
            return format!("Dp1({}, {})", coln, name);
          }
          _ => {
            panic!("Unexpected error.");
          }
        }
      }
      runtime::Core::Dup { eras, glob, expr, body } => {
        let copy = fresh(nams, "cpy");
        let dup0 = fresh(nams, "dp0");
        let dup1 = fresh(nams, "dp1");
        let expr = build_term(book, code, tab, free, vars, nams, lams, dups, expr);
        line(code, tab, &format!("let {} = {};", copy, expr));
        line(code, tab, &format!("let {};", dup0));
        line(code, tab, &format!("let {};", dup1));
        if INLINE_NUMBERS {
          line(code, tab + 0, &format!("if get_tag({}) == U60 || get_tag({}) == F60 {{", copy, copy));
          line(code, tab + 1, "inc_cost(ctx.heap, ctx.tid);");
          line(code, tab + 1, &format!("{} = {};", dup0, copy));
          line(code, tab + 1, &format!("{} = {};", dup1, copy));
          line(code, tab + 0, "} else {");
        }
        let (coln, name) = alloc_dup(code, tab, &mut vec![], nams, dups, *glob);
        if eras.0 {
          line(code, tab + 1, &format!("link(ctx.heap, {} + 0, Era());", name));
        }
        if eras.1 {
          line(code, tab + 1, &format!("link(ctx.heap, {} + 1, Era());", name));
        }
        line(code, tab + 1, &format!("link(ctx.heap, {} + 2, {});", name, copy));
        line(code, tab + 1, &format!("{} = Dp0({}, {});", dup0, coln, name));
        line(code, tab + 1, &format!("{} = Dp1({}, {});", dup1, coln, name));
        if INLINE_NUMBERS {
          line(code, tab + 0, "}");
        }
        vars.push(dup0);
        vars.push(dup1);
        let body = build_term(book, code, tab + 0, free, vars, nams, lams, dups, body);
        vars.pop();
        vars.pop();
        body
      }
      runtime::Core::Sup { val0, val1 } => {
        let name = fresh(nams, "sup");
        let val0 = build_term(book, code, tab, free, vars, nams, lams, dups, val0);
        let val1 = build_term(book, code, tab, free, vars, nams, lams, dups, val1);
        let coln = fresh(nams, "col");
        line(code, tab + 1, &format!("let {} = gen_dup(ctx.heap, ctx.tid);", coln));
        line(code, tab, &format!("let {} = {};", name, alloc_node(free, 2)));
        line(code, tab, &format!("link(ctx.heap, {} + 0, {});", name, val0));
        line(code, tab, &format!("link(ctx.heap, {} + 1, {});", name, val1));
        format!("Sup({}, {})", coln, name)
      }
      runtime::Core::Let { expr, body } => {
        let expr = build_term(book, code, tab, free, vars, nams, lams, dups, expr);
        vars.push(expr);
        let body = build_term(book, code, tab, free, vars, nams, lams, dups, body);
        vars.pop();
        body
      }
      runtime::Core::Lam { eras, glob, body } => {
        let name = alloc_lam(code, tab, free, nams, lams, *glob);
        vars.push(format!("Var({})", name));
        let body = build_term(book, code, tab, free, vars, nams, lams, dups, body);
        vars.pop();
        if *eras {
          line(code, tab, &format!("link(ctx.heap, {} + 0, Era());", name));
        }
        line(code, tab, &format!("link(ctx.heap, {} + 1, {});", name, body));
        format!("Lam({})", name)
      }
      runtime::Core::App { func, argm } => {
        let name = fresh(nams, "app");
        let func = build_term(book, code, tab, free, vars, nams, lams, dups, func);
        let argm = build_term(book, code, tab, free, vars, nams, lams, dups, argm);
        line(code, tab, &format!("let {} = {};", name, alloc_node(free, 2)));
        line(code, tab, &format!("link(ctx.heap, {} + 0, {});", name, func));
        line(code, tab, &format!("link(ctx.heap, {} + 1, {});", name, argm));
        format!("App({})", name)
      }
      runtime::Core::Ctr { func, args } => {
        let cargs: Vec<String> = args.iter().map(|arg| build_term(book, code, tab, free, vars, nams, lams, dups, arg)).collect();
        let name = fresh(nams, "ctr");
        line(code, tab, &format!("let {} = {};", name, alloc_node(free, cargs.len() as u64)));
        for (i, arg) in cargs.iter().enumerate() {
          line(code, tab, &format!("link(ctx.heap, {} + {}, {});", name, i, arg));
        }
        let fnam = build_name(book.id_to_name.get(&func).unwrap_or(&format!("{}", func)));
        format!("Ctr({}, {})", fnam, name)
      }
      runtime::Core::Fun { func, args } => {
        let fargs: Vec<String> = args.iter().map(|arg| build_term(book, code, tab, free, vars, nams, lams, dups, arg)).collect();
        // Inlined U60.if
        if INLINE_NUMBERS && *func == runtime::U60_IF && fargs.len() == 3 {
          let ret = fresh(nams, "ret");
          line(code, tab + 0, &format!("let {};", ret));
          line(code, tab + 0, &format!("if get_tag({}) == U60 {{", fargs[0]));
          line(code, tab + 1, "inc_cost(ctx.heap, ctx.tid);");
          line(code, tab + 1, &format!("if get_num({}) == 0 {{", fargs[0]));
          line(code, tab + 2, &format!("collect(ctx.heap, &ctx.prog.aris, ctx.tid, {});", fargs[1]));
          line(code, tab + 2, &format!("{} = {};", ret, fargs[2]));
          line(code, tab + 1, &format!("}} else {{"));
          line(code, tab + 2, &format!("collect(ctx.heap, &ctx.prog.aris, ctx.tid, {});", fargs[2]));
          line(code, tab + 2, &format!("{} = {};", ret, fargs[1]));
          line(code, tab + 1, &format!("}}"));
          line(code, tab + 0, &format!("}} else {{"));
          let name = fresh(nams, "cal");
          line(code, tab + 1, &format!("let {} = {};", name, alloc_node(free, fargs.len() as u64)));
          for (i, arg) in fargs.iter().enumerate() {
            line(code, tab + 1, &format!("link(ctx.heap, {} + {}, {});", name, i, arg));
          }
          let fnam = build_name(book.id_to_name.get(&func).unwrap_or(&format!("{}", func)));
          line(code, tab + 1, &format!("{} = Fun({}, {})", ret, fnam, name));
          line(code, tab + 0, &format!("}}"));
          return ret;
        // Inlined U60.swap
        } else if INLINE_NUMBERS && *func == runtime::U60_SWAP && fargs.len() == 3 {
          let ret = fresh(nams, "ret");
          line(code, tab + 0, &format!("let {};", ret));
          line(code, tab + 0, &format!("if get_tag({}) == U60 {{", fargs[0]));
          line(code, tab + 1, "inc_cost(ctx.heap, ctx.tid);");
          let both = fresh(nams, "both");
          line(code, tab + 1, &format!("if get_num({}) == 0 {{", fargs[0]));
          line(code, tab + 2, &format!("let {} = {};", both, alloc_node(free, 2)));
          line(code, tab + 2, &format!("link(ctx.heap, {} + 0, {});", both, fargs[1]));
          line(code, tab + 2, &format!("link(ctx.heap, {} + 1, {});", both, fargs[2]));
          line(code, tab + 2, &format!("{} = Ctr(BOTH, {});", ret, both));
          line(code, tab + 1, &format!("}} else {{"));
          line(code, tab + 2, &format!("let {} = {};", both, alloc_node(free, 2)));
          line(code, tab + 2, &format!("link(ctx.heap, {} + 0, {});", both, fargs[2]));
          line(code, tab + 2, &format!("link(ctx.heap, {} + 1, {});", both, fargs[1]));
          line(code, tab + 2, &format!("{} = Ctr(BOTH, {});", ret, both));
          line(code, tab + 1, &format!("}}"));
          line(code, tab + 0, &format!("}} else {{"));
          let name = fresh(nams, "cal");
          line(code, tab + 1, &format!("let {} = {};", name, alloc_node(free, fargs.len() as u64)));
          for (i, arg) in fargs.iter().enumerate() {
            line(code, tab + 1, &format!("link(ctx.heap, {} + {}, {});", name, i, arg));
          }
          let fnam = build_name(book.id_to_name.get(&func).unwrap_or(&format!("{}", func)));
          line(code, tab + 1, &format!("{} = Fun({}, {})", ret, fnam, name));
          line(code, tab + 0, &format!("}}"));
          return ret;
        // Other functions
        } else {
          let name = fresh(nams, "cal");
          line(code, tab, &format!("let {} = {};", name, alloc_node(free, fargs.len() as u64)));
          for (i, arg) in fargs.iter().enumerate() {
            line(code, tab, &format!("link(ctx.heap, {} + {}, {});", name, i, arg));
          }
          let fnam = build_name(book.id_to_name.get(&func).unwrap_or(&format!("{}", func)));
          return format!("Fun({}, {})", fnam, name);
        }
      }
      runtime::Core::U6O { numb } => {
        format!("U6O({})", numb)
      }
      runtime::Core::F6O { numb } => {
        format!("F6O({})", numb)
      }
      runtime::Core::Op2 { oper, val0, val1 } => {
        let retx = fresh(nams, "ret");
        let name = fresh(nams, "op2");
        let val0 = build_term(book, code, tab, free, vars, nams, lams, dups, val0);
        let val1 = build_term(book, code, tab, free, vars, nams, lams, dups, val1);
        line(code, tab + 0, &format!("let {};", retx));
        // Optimization: do inline operation, avoiding Op2 allocation, when operands are already number
        if INLINE_NUMBERS {
          line(code, tab + 0, &format!("if get_tag({}) == U60 && get_tag({}) == U60 {{", val0, val1));
          let a = format!("get_num({})", val0);
          let b = format!("get_num({})", val1);
          match *oper {
            runtime::ADD => line(code, tab + 1, &format!("{} = U6O(u60::add({}, {}));", retx, a, b)),
            runtime::SUB => line(code, tab + 1, &format!("{} = U6O(u60::sub({}, {}));", retx, a, b)),
            runtime::MUL => line(code, tab + 1, &format!("{} = U6O(u60::mul({}, {}));", retx, a, b)),
            runtime::DIV => line(code, tab + 1, &format!("{} = U6O(u60::div({}, {}));", retx, a, b)),
            runtime::MOD => line(code, tab + 1, &format!("{} = U6O(u60::mdl({}, {}));", retx, a, b)),
            runtime::AND => line(code, tab + 1, &format!("{} = U6O(u60::and({}, {}));", retx, a, b)),
            runtime::OR  => line(code, tab + 1, &format!("{} = U6O(u60::or({}, {}));", retx, a, b)),
            runtime::XOR => line(code, tab + 1, &format!("{} = U6O(u60::xor({}, {}));", retx, a, b)),
            runtime::SHL => line(code, tab + 1, &format!("{} = U6O(u60::shl({}, {}));", retx, a, b)),
            runtime::SHR => line(code, tab + 1, &format!("{} = U6O(u60::shr({}, {}));", retx, a, b)),
            runtime::LTN => line(code, tab + 1, &format!("{} = U6O(u60::ltn({}, {}));", retx, a, b)),
            runtime::LTE => line(code, tab + 1, &format!("{} = U6O(u60::lte({}, {}));", retx, a, b)),
            runtime::EQL => line(code, tab + 1, &format!("{} = U6O(u60::eql({}, {}));", retx, a, b)),
            runtime::GTE => line(code, tab + 1, &format!("{} = U6O(u60::gte({}, {}));", retx, a, b)),
            runtime::GTN => line(code, tab + 1, &format!("{} = U6O(u60::gtn({}, {}));", retx, a, b)),
            runtime::NEQ => line(code, tab + 1, &format!("{} = U6O(u60::neq({}, {}));", retx, a, b)),
            _            => line(code, tab + 1, &format!("{} = 0;", retx)),
          }
          line(code, tab + 1, "inc_cost(ctx.heap, ctx.tid);");
          line(code, tab + 0, &format!("}} else if get_tag({}) == F60 && get_tag({}) == F60 {{", val0, val1));
          let a = format!("get_num({})", val0);
          let b = format!("get_num({})", val1);
          match *oper {
            runtime::ADD => line(code, tab + 1, &format!("{} = F6O(f60::add({}, {}));", retx, a, b)),
            runtime::SUB => line(code, tab + 1, &format!("{} = F6O(f60::sub({}, {}));", retx, a, b)),
            runtime::MUL => line(code, tab + 1, &format!("{} = F6O(f60::mul({}, {}));", retx, a, b)),
            runtime::DIV => line(code, tab + 1, &format!("{} = F6O(f60::div({}, {}));", retx, a, b)),
            runtime::MOD => line(code, tab + 1, &format!("{} = F6O(f60::mdl({}, {}));", retx, a, b)),
            runtime::AND => line(code, tab + 1, &format!("{} = F6O(f60::and({}, {}));", retx, a, b)),
            runtime::OR  => line(code, tab + 1, &format!("{} = F6O(f60::or({}, {}));", retx, a, b)),
            runtime::XOR => line(code, tab + 1, &format!("{} = F6O(f60::xor({}, {}));", retx, a, b)),
            runtime::SHL => line(code, tab + 1, &format!("{} = F6O(f60::shl({}, {}));", retx, a, b)),
            runtime::SHR => line(code, tab + 1, &format!("{} = F6O(f60::shr({}, {}));", retx, a, b)),
            runtime::LTN => line(code, tab + 1, &format!("{} = F6O(f60::ltn({}, {}));", retx, a, b)),
            runtime::LTE => line(code, tab + 1, &format!("{} = F6O(f60::lte({}, {}));", retx, a, b)),
            runtime::EQL => line(code, tab + 1, &format!("{} = F6O(f60::eql({}, {}));", retx, a, b)),
            runtime::GTE => line(code, tab + 1, &format!("{} = F6O(f60::gte({}, {}));", retx, a, b)),
            runtime::GTN => line(code, tab + 1, &format!("{} = F6O(f60::gtn({}, {}));", retx, a, b)),
            runtime::NEQ => line(code, tab + 1, &format!("{} = F6O(f60::neq({}, {}));", retx, a, b)),
            _            => line(code, tab + 1, &format!("{} = 0;", retx)),
          }
          line(code, tab + 1, "inc_cost(ctx.heap, ctx.tid);");
          line(code, tab + 0, "} else {");
        }
        line(code, tab + 1, &format!("let {} = {};", name, alloc_node(&mut vec![], 2)));
        line(code, tab + 1, &format!("link(ctx.heap, {} + 0, {});", name, val0));
        line(code, tab + 1, &format!("link(ctx.heap, {} + 1, {});", name, val1));
        let oper_name = match *oper {
          runtime::ADD => "ADD",
          runtime::SUB => "SUB",
          runtime::MUL => "MUL",
          runtime::DIV => "DIV",
          runtime::MOD => "MOD",
          runtime::AND => "AND",
          runtime::OR  => "OR",
          runtime::XOR => "XOR",
          runtime::SHL => "SHL",
          runtime::SHR => "SHR",
          runtime::LTN => "LTN",
          runtime::LTE => "LTE",
          runtime::EQL => "EQL",
          runtime::GTE => "GTE",
          runtime::GTN => "GTN",
          runtime::NEQ => "NEQ",
          _            => "?",
        };
        line(code, tab + 1, &format!("{} = Op2({}, {});", retx, oper_name, name));
        if INLINE_NUMBERS {
          line(code, tab + 0, "}");
        }
        retx
      }
    }
  }
  fn fresh(nams: &mut u64, name: &str) -> String {
    let name = format!("{}_{}", name, nams);
    *nams += 1;
    name
  }
  let mut nams = 0;
  let mut vars : Vec<String> = vec![];
  for runtime::RuleVar { param, field, erase: _ } in rvrs {
    match field {
      Some(field) => {
        line(code, tab + 0, &format!("let arg{}_{} = load_arg(ctx.heap, arg{}, {});", param, field, param, field));
        vars.push(format!("arg{}_{}", param, field));
      }
      None => {
        vars.push(format!("arg{}", param));
      }
    }
  }
  let mut lams: HashMap<u64, String> = HashMap::new();
  let mut dups: HashMap<u64, (String,String)> = HashMap::new();
  build_term(book, code, tab, free, &mut vars, &mut nams, &mut lams, &mut dups, term)
}

pub fn line(code: &mut String, tab: u64, line: &str) {
  for _ in 0..tab {
    code.push_str("  ");
  }
  code.push_str(line);
  code.push('\n');
}

pub fn get_var(var: &runtime::RuleVar) -> String {
  let runtime::RuleVar { param, field, erase: _ } = var;
  match field {
    Some(i) => { format!("arg{}_{}", param, i) }
    None => { format!("arg{}", param) }
  }
}
