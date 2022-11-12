#![allow(unreachable_code)]
#![allow(clippy::identity_op)]

use std::collections::HashMap;

use crate::language as language;
use crate::runtime as runtime;

pub fn compile(code: &str, file_name: &str) -> std::io::Result<()> {

  // hvm
  std::fs::create_dir("./_hvm_").ok();
  std::fs::write("./_hvm_/Cargo.toml", CARGO_TOML)?;

  // hvm/src
  std::fs::create_dir("./_hvm_/src").ok();
  std::fs::write("./_hvm_/src/main.rs", MAIN_RS)?;

  // hvm/src/language
  std::fs::create_dir("./_hvm_/src/language").ok();
  std::fs::write("./_hvm_/src/language/mod.rs"      , include_str!("./../language/mod.rs"))?;
  std::fs::write("./_hvm_/src/language/parser.rs"   , include_str!("./../language/parser.rs"))?;
  std::fs::write("./_hvm_/src/language/readback.rs" , include_str!("./../language/readback.rs"))?;
  std::fs::write("./_hvm_/src/language/rulebook.rs" , include_str!("./../language/rulebook.rs"))?;
  std::fs::write("./_hvm_/src/language/syntax.rs"   , include_str!("./../language/syntax.rs"))?;

  // hvm/src/runtime
  std::fs::create_dir("./_hvm_/src/runtime").ok();
  std::fs::write("./_hvm_/src/runtime/mod.rs", include_str!("./../runtime/mod.rs"))?;

  // hvm/src/runtime/base
  let (precomp_rs, reducer_rs) = compile_code(code).unwrap();
  std::fs::create_dir("./_hvm_/src/runtime/base").ok();
  std::fs::write("./_hvm_/src/runtime/base/mod.rs"     , include_str!("./../runtime/base/mod.rs"))?;
  std::fs::write("./_hvm_/src/runtime/base/debug.rs"   , include_str!("./../runtime/base/debug.rs"))?;
  std::fs::write("./_hvm_/src/runtime/base/memory.rs"  , include_str!("./../runtime/base/memory.rs"))?;
  std::fs::write("./_hvm_/src/runtime/base/precomp.rs" , precomp_rs)?;
  std::fs::write("./_hvm_/src/runtime/base/program.rs" , include_str!("./../runtime/base/program.rs"))?;
  std::fs::write("./_hvm_/src/runtime/base/reducer.rs" , reducer_rs)?;

  // hvm/src/runtime/data
  std::fs::create_dir("./_hvm_/src/runtime/data").ok();
  std::fs::write("./_hvm_/src/runtime/data/mod.rs"         , include_str!("./../runtime/data/mod.rs"))?;
  std::fs::write("./_hvm_/src/runtime/data/allocator.rs"   , include_str!("./../runtime/data/allocator.rs"))?;
  std::fs::write("./_hvm_/src/runtime/data/redex_bag.rs"   , include_str!("./../runtime/data/redex_bag.rs"))?;
  std::fs::write("./_hvm_/src/runtime/data/u64_map.rs"     , include_str!("./../runtime/data/u64_map.rs"))?;
  std::fs::write("./_hvm_/src/runtime/data/visit_queue.rs" , include_str!("./../runtime/data/visit_queue.rs"))?;

  // hvm/src/runtime/rule
  std::fs::create_dir("./_hvm_/src/runtime/rule").ok();
  std::fs::write("./_hvm_/src/runtime/rule/mod.rs" , include_str!("./../runtime/rule/mod.rs"))?;
  std::fs::write("./_hvm_/src/runtime/rule/app.rs" , include_str!("./../runtime/rule/app.rs"))?;
  std::fs::write("./_hvm_/src/runtime/rule/dup.rs" , include_str!("./../runtime/rule/dup.rs"))?;
  std::fs::write("./_hvm_/src/runtime/rule/fun.rs" , include_str!("./../runtime/rule/fun.rs"))?;
  std::fs::write("./_hvm_/src/runtime/rule/op2.rs" , include_str!("./../runtime/rule/op2.rs"))?;

  return Ok(());
}

fn compile_name(name: &str) -> String {
  // TODO: this can still cause some name collisions.
  // Note: avoiding the use of `$` because it is not an actually valid
  // identifier character in C.
  //let name = name.replace('_', "__");
  let name = name.replace('.', "_").replace('$', "_S_");
  format!("_{}_", name)
}

fn compile_code(code: &str) -> Result<(String,String), String> {
  let file = language::syntax::read_file(code)?;
  let book = language::rulebook::gen_rulebook(&file);
  runtime::gen_functions(&book);
  Ok(compile_rulebook(&book))
}

fn compile_rulebook(book: &language::rulebook::RuleBook) -> (String, String) {
  // precomp ids
  let mut precomp_ids = String::new();
  for (id, name) in itertools::sorted(book.id_to_name.iter()) {
    if id >= &runtime::PRECOMP_COUNT {
      line(&mut precomp_ids, 0, &format!("pub const {} : u64 = {};", &compile_name(name), id));
    }
  }

  // precomp els
  let mut precomp_els = String::new();
  for id in itertools::sorted(book.id_to_arit.keys()) {
    if id >= &runtime::PRECOMP_COUNT {
      let name = book.id_to_name.get(id).unwrap();
      let comp = &compile_name(name);
      let arity = book.id_to_arit.get(id).unwrap();
      line(&mut precomp_els, 0, &format!(r#"  Precomp {{"#));
      line(&mut precomp_els, 0, &format!(r#"    id    : {},"#, comp));
      line(&mut precomp_els, 0, &format!(r#"    name  : "{}","#, &name));
      line(&mut precomp_els, 0, &format!(r#"    arity : {},"#, arity));
      if *book.ctr_is_cal.get(name).unwrap_or(&false) {
        line(&mut precomp_els, 0, &format!(r#"    funcs : Some(PrecompFns {{"#));
        line(&mut precomp_els, 0, &format!(r#"      visit: {}_visit,"#, comp));
        line(&mut precomp_els, 0, &format!(r#"      apply: {}_apply,"#, comp));
        line(&mut precomp_els, 0, &format!(r#"    }}),"#));
      } else {
        line(&mut precomp_els, 0, &format!(r#"    funcs : None,"#));
      }
      line(&mut precomp_els, 0, &format!(r#"  }},"#));
    }
  }

  // precomp fns
  let mut precomp_fns = String::new();
  for id in itertools::sorted(book.id_to_arit.keys()) {
    if id >= &runtime::PRECOMP_COUNT {
      let name = book.id_to_name.get(id).unwrap();
      if let Some(rules) = book.rule_group.get(name) {
        let (visit_fn, apply_fn) = compile_function(book, &name, &rules.1);
        line(&mut precomp_fns, 0, &format!("{}", visit_fn));
        line(&mut precomp_fns, 0, &format!("{}", apply_fn));
      }
    }
  }

  // fast visit
  let mut fast_visit = String::new();
  line(&mut fast_visit, 7, &format!("match fid {{"));
  for id in itertools::sorted(book.id_to_arit.keys()) {
    if id >= &runtime::PRECOMP_COUNT {
      let name = book.id_to_name.get(id).unwrap();
      if let Some(rules) = book.rule_group.get(name) {
        let (visit_fun, apply_fun) = compile_function(book, &name, &rules.1);
        line(&mut fast_visit, 8, &format!("{} => {{", &compile_name(&name)));
        line(&mut fast_visit, 9, &format!("if {}_visit(ReduceCtx {{ heap, prog, tid, term, visit, redex, cont: &mut cont, host: &mut host }}) {{", &compile_name(&name)));
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
  for id in itertools::sorted(book.id_to_arit.keys()) {
    if id >= &runtime::PRECOMP_COUNT {
      let name = book.id_to_name.get(id).unwrap();
      let rules = 
      if let Some(rules) = book.rule_group.get(name) {
        let (visit_fun, apply_fun) = compile_function(book, &name, &rules.1);
        line(&mut fast_apply, 9, &format!("{} => {{", &compile_name(&name)));
        line(&mut fast_apply, 10, &format!("if {}_apply(ReduceCtx {{ heap, prog, tid, term, visit, redex, cont: &mut cont, host: &mut host }}) {{", &compile_name(&name)));
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

fn compile_function(
  book  : &language::rulebook::RuleBook,
  fname : &str,
  rules : &[language::syntax::Rule],
) -> (String, String) {
  if let runtime::Function::Interpreted {
    arity: fn_arity,
    visit: fn_visit,
    apply: fn_apply,
  } = runtime::build_function(book, fname, rules) {

    // Visit
    // -----

    let mut visit = String::new();
    line(&mut visit, 0, &format!("#[inline(always)]"));
    line(&mut visit, 0, &format!("pub fn {}_visit(ctx: ReduceCtx) -> bool {{", &compile_name(fname)));
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
        line(&mut visit, 3, &format!("ctx.visit.push(new_visit(unsafe {{ vbuf.get_unchecked({}).load(Ordering::Relaxed) }}, goup));", i));
        line(&mut visit, 2, &format!("}}"));
      }
      line(&mut visit, 2, &format!("*ctx.cont = goup;"));
      line(&mut visit, 2, &format!("*ctx.host = unsafe {{ vbuf.get_unchecked(vlen - 1).load(Ordering::Relaxed) }};"));
      line(&mut visit, 2, &format!("return true;"));
      line(&mut visit, 1, &format!("}}"));
    }
    line(&mut visit, 0, &format!("}}"));

    //OLD_VISITER:
    //let mut visit = String::new();
    //line(&mut visit, 0, &format!("#[inline(always)]"));
    //line(&mut visit, 0, &format!("pub fn {}_visit(ctx: ReduceCtx) -> bool {{", &compile_name(fname)));
    //if fn_visit.strict_idx.is_empty() {
      //line(&mut visit, 1, "return false;");
    //} else {
      //line(&mut visit, 1, &format!("let goup = ctx.redex.insert(ctx.tid, new_redex(*ctx.host, *ctx.cont, {}));", fn_visit.strict_idx.len()));
      //for (i, strict) in fn_visit.strict_idx.iter().enumerate() {
        //if i < fn_visit.strict_idx.len() - 1 {
          //line(&mut visit, 1, &format!("ctx.visit.push(new_visit(get_loc(ctx.term, {}), goup));", strict));
        //} else {
          //line(&mut visit, 1, &format!("*ctx.cont = goup;"));
          //line(&mut visit, 1, &format!("*ctx.host = get_loc(ctx.term, {});", strict));
          //line(&mut visit, 1, &format!("return true;"));
        //}
      //}
    //}
    //line(&mut visit, 0, "}");

    // Apply
    // -----
    
    let mut apply = String::new();
    
    line(&mut apply, 0, &format!("#[inline(always)]"));
    line(&mut apply, 0, &format!("pub fn {}_apply(ctx: ReduceCtx) -> bool {{", &compile_name(fname)));

    // Loads strict arguments
    for i in 0 .. fn_arity {
      line(&mut apply, 1, &format!("let arg{} = load_arg(ctx.heap, ctx.term, {});", i, i));
    }

    // Applies the fun_sup rule to superposed args
    for (i, is_strict) in fn_visit.strict_map.iter().enumerate() {
      if *is_strict {
        line(&mut apply, 1, &format!("if get_tag(arg{}) == SUP {{", i));
        line(&mut apply, 2, &format!("fun::superpose(ctx.heap, &ctx.prog.arit, ctx.tid, *ctx.host, ctx.term, arg{}, {});", i, i));
        line(&mut apply, 1, "}");
      }
    }

    // For each rule condition vector
    for (r, rule) in fn_apply.rules.iter().enumerate() {
      let mut matched: Vec<String> = Vec::new();

      // Tests each rule condition (ex: `get_tag(args[0]) == SUCC`)
      for (i, cond) in rule.cond.iter().enumerate() {
        let i = i as u64;
        if runtime::get_tag(*cond) == runtime::NUM {
          let same_tag = format!("get_tag(arg{}) == NUM", i);
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
            let is_num
              = format!("get_tag(arg{}) == NUM", i);

            // Matches constructor labels
            let is_ctr = format!("({} && {})",
              format!("get_tag(arg{}) == CTR", i),
              format!("arity_of(heap, arg{}) == 0u", i));

            // Matches HOAS numbers and constructors
            let is_hoas_ctr_num = format!("({} && {} && {})",
              format!("get_tag(arg{}) == CTR", i),
              format!("get_ext(arg{}) >= HOAS_CT0", i),
              format!("get_ext(arg{}) <= HOAS_NUM", i));

            matched.push(format!("({} || {} || {})", is_num, is_ctr, is_hoas_ctr_num));

          // Only match default variables on CTRs and NUMs
          } else {
            let is_ctr = format!("get_tag(arg{}) == CTR", i);
            let is_num = format!("get_tag(arg{}) == NUM", i);
            matched.push(format!("({} || {})", is_ctr, is_num));
          }

        }
      }

      let conds = if matched.is_empty() { String::from("true") } else { matched.join(" && ") };
      line(&mut apply, 1, &format!("if {} {{", conds));

      // Increments the gas count
      line(&mut apply, 2, "inc_cost(ctx.heap, ctx.tid);");

      // Builds the right-hand side term (ex: `(Succ (Add a b))`)
      //let done = compile_function_rule_body(&mut apply, 2, &rule.body, &rule.vars);
      let done = compile_function_rule_rhs(book, &mut apply, 2, &rule.core, &rule.vars);
      line(&mut apply, 2, &format!("let done = {};", done));

      // Links the host location to it
      line(&mut apply, 2, "link(ctx.heap, *ctx.host, done);");

      // Collects unused variables (none in this example)
      for dynvar @ runtime::RuleVar { param: _, field: _, erase } in rule.vars.iter() {
        if *erase {
          line(&mut apply, 2, &format!("collect(ctx.heap, &ctx.prog.arit, ctx.tid, {});", get_var(dynvar)));
        }
      }

      // Clears the matched ctrs (the `(Succ ...)` and the `(Add ...)` ctrs)
      for (i, arity) in &rule.free {
        let i = *i as u64;
        line(&mut apply, 2, &format!("free(ctx.heap, ctx.tid, get_loc(arg{}, 0), {});", i, arity));
      }
      line(&mut apply, 2, &format!("free(ctx.heap, ctx.tid, get_loc(ctx.term, 0), {});", fn_visit.strict_map.len()));
      line(&mut apply, 2, "return true;");
      line(&mut apply, 1, "}");
    }
    line(&mut apply, 1, "return false;");

    line(&mut apply, 0, "}");

    (visit, apply)
  } else {
    panic!("Unexpected error.");
  }
}

fn compile_function_rule_rhs(
  book : &language::rulebook::RuleBook,
  code : &mut String,
  tab  : u64,
  term : &runtime::Core,
  vars : &[runtime::RuleVar],
) -> String {
  fn alloc_lam(code: &mut String, tab: u64, nams: &mut u64, lams: &mut HashMap<u64, String>, glob: u64) -> String {
    if let Some(got) = lams.get(&glob) {
      got.clone()
    } else {
      let name = fresh(nams, "lam");
      line(code, tab, &format!("let {} = alloc(ctx.heap, ctx.tid, 2);", name));
      if glob != 0 {
        // FIXME: sanitizer still can't detect if a scopeless lambda doesn't use its bound
        // variable, so we must write an Era() here. When it does, we can remove this line.
        line(code, tab, &format!("link(heap, {} + 0, Era());", name));
        lams.insert(glob, name.clone());
      }
      name
    }
  }
  fn alloc_dup(code: &mut String, tab: u64, nams: &mut u64, dups: &mut HashMap<u64, (String,String)>, glob: u64) -> (String, String) {
    if let Some(got) = dups.get(&glob) {
      return got.clone();
    } else {
      let coln = fresh(nams, "col");
      let name = fresh(nams, "dup");
      line(code, tab + 1, &format!("let {} = gen_dup(ctx.heap, ctx.tid);", coln));
      line(code, tab + 1, &format!("let {} = alloc(ctx.heap, ctx.tid, 3);", name));
      if glob != 0 {
        line(code, tab, &format!("link(ctx.heap, {} + 0, Era());", name)); // FIXME: remove when possible (same as above)
        line(code, tab, &format!("link(ctx.heap, {} + 1, Era());", name)); // FIXME: remove when possible (same as above)
        dups.insert(glob, (coln.clone(), name.clone()));
      }
      return (coln, name);
    }
  }
  fn compile_term(
    book : &language::rulebook::RuleBook,
    code : &mut String,
    tab  : u64,
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
            return format!("Var({})", alloc_lam(code, tab, nams, lams, *glob));
          }
          runtime::DP0 => {
            let (coln, name) = alloc_dup(code, tab, nams, dups, *glob);
            return format!("Dp0({}, {})", coln, name);
          }
          runtime::DP1 => {
            let (coln, name) = alloc_dup(code, tab, nams, dups, *glob);
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
        let expr = compile_term(book, code, tab, vars, nams, lams, dups, expr);
        line(code, tab, &format!("let {} = {};", copy, expr));
        line(code, tab, &format!("let {};", dup0));
        line(code, tab, &format!("let {};", dup1));
        if INLINE_NUMBERS {
          line(code, tab + 0, &format!("if get_tag({}) == NUM {{", copy));
          line(code, tab + 1, "inc_cost(ctx.heap, ctx.tid);");
          line(code, tab + 1, &format!("{} = {};", dup0, copy));
          line(code, tab + 1, &format!("{} = {};", dup1, copy));
          line(code, tab + 0, "} else {");
        }
        let (coln, name) = alloc_dup(code, tab, nams, dups, *glob);
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
        let body = compile_term(book, code, tab + 0, vars, nams, lams, dups, body);
        vars.pop();
        vars.pop();
        body
      }
      runtime::Core::Sup { val0, val1 } => {
        let name = fresh(nams, "sup");
        let val0 = compile_term(book, code, tab, vars, nams, lams, dups, val0);
        let val1 = compile_term(book, code, tab, vars, nams, lams, dups, val1);
        let coln = fresh(nams, "col");
        line(code, tab + 1, &format!("let {} = gen_dup(ctx.heap, ctx.tid);", coln));
        line(code, tab, &format!("let {} = alloc(ctx.heap, ctx.tid, 2);", name));
        line(code, tab, &format!("link(ctx.heap, {} + 0, {});", name, val0));
        line(code, tab, &format!("link(ctx.heap, {} + 1, {});", name, val1));
        format!("Sup({}, {})", coln, name)
      }
      runtime::Core::Let { expr, body } => {
        let expr = compile_term(book, code, tab, vars, nams, lams, dups, expr);
        vars.push(expr);
        let body = compile_term(book, code, tab, vars, nams, lams, dups, body);
        vars.pop();
        body
      }
      runtime::Core::Lam { eras, glob, body } => {
        let name = alloc_lam(code, tab, nams, lams, *glob);
        vars.push(format!("Var({})", name));
        let body = compile_term(book, code, tab, vars, nams, lams, dups, body);
        vars.pop();
        if *eras {
          line(code, tab, &format!("link(ctx.heap, {} + 0, Era());", name));
        }
        line(code, tab, &format!("link(ctx.heap, {} + 1, {});", name, body));
        format!("Lam({})", name)
      }
      runtime::Core::App { func, argm } => {
        let name = fresh(nams, "app");
        let func = compile_term(book, code, tab, vars, nams, lams, dups, func);
        let argm = compile_term(book, code, tab, vars, nams, lams, dups, argm);
        line(code, tab, &format!("let {} = alloc(ctx.heap, ctx.tid, 2);", name));
        line(code, tab, &format!("link(ctx.heap, {} + 0, {});", name, func));
        line(code, tab, &format!("link(ctx.heap, {} + 1, {});", name, argm));
        format!("App({})", name)
      }
      runtime::Core::Ctr { func, args } => {
        let ctr_args: Vec<String> = args.iter().map(|arg| compile_term(book, code, tab, vars, nams, lams, dups, arg)).collect();
        let name = fresh(nams, "ctr");
        line(code, tab, &format!("let {} = alloc(ctx.heap, ctx.tid, {});", name, ctr_args.len()));
        for (i, arg) in ctr_args.iter().enumerate() {
          line(code, tab, &format!("link(ctx.heap, {} + {}, {});", name, i, arg));
        }
        let fnam = compile_name(book.id_to_name.get(&func).unwrap_or(&format!("{}", func)));
        format!("Ctr({}, {})", fnam, name)
      }
      runtime::Core::Fun { func, args } => {
        let cal_args: Vec<String> = args.iter().map(|arg| compile_term(book, code, tab, vars, nams, lams, dups, arg)).collect();
        let name = fresh(nams, "cal");
        line(code, tab, &format!("let {} = alloc(ctx.heap, ctx.tid, {});", name, cal_args.len()));
        for (i, arg) in cal_args.iter().enumerate() {
          line(code, tab, &format!("link(ctx.heap, {} + {}, {});", name, i, arg));
        }
        let fnam = compile_name(book.id_to_name.get(&func).unwrap_or(&format!("{}", func)));
        format!("Fun({}, {})", fnam, name)
      }
      runtime::Core::Num { numb } => {
        format!("Num({})", numb)
      }
      runtime::Core::Op2 { oper, val0, val1 } => {
        let retx = fresh(nams, "ret");
        let name = fresh(nams, "op2");
        let val0 = compile_term(book, code, tab, vars, nams, lams, dups, val0);
        let val1 = compile_term(book, code, tab, vars, nams, lams, dups, val1);
        line(code, tab + 0, &format!("let {};", retx));
        // Optimization: do inline operation, avoiding Op2 allocation, when operands are already number
        if INLINE_NUMBERS {
          line(
            code,
            tab + 0,
            &format!("if get_tag({}) == NUM && get_tag({}) == NUM {{", val0, val1),
          );
          let a = format!("get_num({})", val0);
          let b = format!("get_num({})", val1);
          match *oper {
            runtime::ADD => line(code, tab + 1, &format!("{} = Num({} + {});", retx, a, b)),
            runtime::SUB => line(code, tab + 1, &format!("{} = Num({} - {});", retx, a, b)),
            runtime::MUL => line(code, tab + 1, &format!("{} = Num({} * {});", retx, a, b)),
            runtime::DIV => line(code, tab + 1, &format!("{} = Num({} / {});", retx, a, b)),
            runtime::MOD => line(code, tab + 1, &format!("{} = Num({} % {});", retx, a, b)),
            runtime::AND => line(code, tab + 1, &format!("{} = Num({} & {});", retx, a, b)),
            runtime::OR  => line(code, tab + 1, &format!("{} = Num({} | {});", retx, a, b)),
            runtime::XOR => line(code, tab + 1, &format!("{} = Num({} ^ {});", retx, a, b)),
            runtime::SHL => line(code, tab + 1, &format!("{} = Num({} << {});", retx, a, b)),
            runtime::SHR => line(code, tab + 1, &format!("{} = Num({} >> {});", retx, a, b)),
            runtime::LTN => line(code, tab + 1, &format!("{} = Num(if {} < {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            runtime::LTE => line(code, tab + 1, &format!("{} = Num(if {} <= {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            runtime::EQL => line(code, tab + 1, &format!("{} = Num(if {} == {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            runtime::GTE => line(code, tab + 1, &format!("{} = Num(if {} >= {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            runtime::GTN => line(code, tab + 1, &format!("{} = Num(if {} >  {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            runtime::NEQ => line(code, tab + 1, &format!("{} = Num(if {} != {} {{ 1 }} else {{ 0 }});", retx, a, b)),
            _ => line(code, tab + 1, &format!("{} = ?;", retx)),
          }
          line(code, tab + 1, "inc_cost(ctx.heap, ctx.tid);");
          line(code, tab + 0, "} else {");
        }
        line(code, tab + 1, &format!("let {} = alloc(ctx.heap, ctx.tid, 2);", name));
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
  let mut vars: Vec<String> = vars
    .iter()
    .map(|_var @ runtime::RuleVar { param, field, erase: _ }| match field {
      Some(field) => {
        format!("load_arg(ctx.heap, arg{}, {})", param, field)
      }
      None => {
        format!("arg{}", param)
      }
    })
    .collect();
  let mut lams: HashMap<u64, String> = HashMap::new();
  let mut dups: HashMap<u64, (String,String)> = HashMap::new();
  compile_term(book, code, tab, &mut vars, &mut nams, &mut lams, &mut dups, term)
}

fn line(code: &mut String, tab: u64, line: &str) {
  for _ in 0..tab {
    code.push_str("  ");
  }
  code.push_str(line);
  code.push('\n');
}

fn get_var(var: &runtime::RuleVar) -> String {
  let runtime::RuleVar { param, field, erase: _ } = var;
  match field {
    Some(i) => {
      format!("load_arg(ctx.heap, arg{}, {})", param, i)
    }
    None => {
      format!("arg{}", param)
    }
  }
}

const CARGO_TOML : &str = r#"
[package]
name = "hvm-app"
version = "0.1.0"
edition = "2021"
description = "An HVM application"
repository = "https://github.com/Kindelia/HVM"
license = "MIT"
keywords = ["functional", "language", "runtime", "compiler", "target"]
categories = ["compilers"]

[[bin]]
name = "hvm-app"
test = false

[dependencies]
crossbeam = "0.8.2"
thread-priority = "0.9.2"
itertools = "0.10"
num_cpus = "1.13"
regex = "1.5.4"
fastrand = "1.8.0"
highlight_error = "0.1.1"
clap = { version = "3.1.8", features = ["derive"] }
wasm-bindgen = "0.2.82"
reqwest = { version = "0.11.11", features = ["blocking"] }
web-sys = { version = "0.3", features = ["console"] }
instant = { version = "0.1", features = [ "wasm-bindgen", "inaccurate" ] }
"#;

const MAIN_RS : &str = r#"
#![feature(atomic_from_mut)]
#![feature(atomic_mut_ptr)]
#![allow(non_upper_case_globals)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_macros)]
#![allow(unused_parens)]
#![allow(unused_labels)]

mod language;
mod runtime;

fn make_main_call(params: &Vec<String>) -> Result<language::syntax::Term, String> {
  let name = "Main".to_string();
  let args = params.iter().map(|x| language::syntax::read_term(x).unwrap()).collect();
  return Ok(language::syntax::Term::Ctr { name, args });
}

fn run_code(code: &str, debug: bool, params: Vec<String>, size: usize) -> Result<(), String> {
  let call = make_main_call(&params)?;
  let (norm, cost, size, time) = runtime::eval_code(&call, code, debug, size)?;
  println!("{}", norm);
  eprintln!();
  eprintln!("Rewrites: {} ({:.2} MR/s)", cost, (cost as f64) / (time as f64) / 1000.0);
  eprintln!("Mem.Size: {}", size);
  return Ok(());
}

fn main() -> Result<(), String> {
  let params : Vec<String> = vec![];
  let size = runtime::HEAP_SIZE;
  let debug = false;
  run_code("", debug, params, size)?;
  return Ok(());
}
"#;

