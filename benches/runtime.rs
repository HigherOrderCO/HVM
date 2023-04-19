use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn radix_sort(c: &mut Criterion) {
  let thread_count = 1;
  let code = include_str!("../examples/sort/radix/main.hvm");
  let file = hvm::syntax::read_file(code).unwrap();

  // Parses and reads the input file
  let file = hvm::syntax::read_file(&format!("{}\nHVM_MAIN_CALL = (Main 16)", file)).unwrap();
  // Converts the file to a Rulebook
  let book = hvm::rulebook::gen_rulebook(&file);

  // Creates the runtime program
  let mut prog = hvm::Program::new();

  // Adds the interpreted functions (from the Rulebook)
  prog.add_book(&book);

  // Creates the runtime heap
  let heap = hvm::new_heap(hvm::default_heap_size(), thread_count);
  let tids = hvm::new_tids(thread_count);

  hvm::link(&heap, 0, hvm::Fun(*book.name_to_id.get("HVM_MAIN_CALL").unwrap(), 0));
  let host = 0;

  c.bench_function("radix_sort, serial", |b| {
    b.iter(|| {
        // Allocates the main term
        hvm::normalize(&heap, &prog, &tids[..1], black_box(host), false);
    })
  });

  c.bench_function("radix_sort, parallel", |b| {
    b.iter(|| {
        // Allocates the main term
        hvm::normalize(&heap, &prog, &tids, black_box(host), false);
    })
  });

  // Frees used memory
  hvm::collect(&heap, &prog.aris, tids[0], hvm::load_ptr(&heap, host));
  hvm::free(&heap, 0, 0, 1);
}

criterion_group!(benches, radix_sort);
criterion_main!(benches);
