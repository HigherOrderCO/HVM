use hovm::builder;
use pprof::protos::Message;
use std::{fs::File, io::Write};

fn main() {
  let guard = pprof::ProfilerGuard::new(100).unwrap();
  let code = "
    //(Main) = (λf λx (f (f x)) λf λx (f (f x)))
    (Slow (Z))      = 1
    (Slow (S pred)) = (+ (Slow pred) (Slow pred))
    
    (Main) = (Slow (S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (S(S(S(S (Z) )))) )))) )))) )))) )))) )))) )) )
  ";
  let (norm, cost, size, time) = builder::eval_code("Main", code);
  println!("Rewrites: {} ({:.2} MR/s)", cost, (cost as f64) / (time as f64) / 1000.0);
  println!("Mem.Size: {}", size);
  println!();
  println!("{}", norm);
  if let Ok(report) = guard.report().build() {
    // Output flamegraph.
    let file = File::create("flamegraph.svg").unwrap();
    report.flamegraph(file).unwrap();

    // Output protobuf.
    let mut file = File::create("profile.pb").unwrap();
    let profile = report.pprof().unwrap();

    let mut content = Vec::new();
    profile.encode(&mut content).unwrap();
    file.write_all(&content).unwrap();
  };
}
