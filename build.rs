fn main() {
  let target = std::env::var("TARGET").unwrap();
  // This is is part of a larger workaround to support compiling on stable rust.
  //
  // This is the set of platforms that have been manually verified to have atomic integers with the
  // same alignment as the normal integers. There are doubtless platforms that satisfy the
  // requirement that aren't in the list, and should be added if anybody needs to compile for the
  // platform. Pull requests welcome.
  //
  // This is a workaround for the unstable `cfg(target_has_atomic_equal_alignment = "n")` check, and
  // should be removed once that or the `atomic_from_mut` and `atomic_mut_ptr` features are stable.
  //
  // See also `polyfills.rs`.
  //
  // See
  // https://users.rust-lang.org/t/is-there-a-way-to-emulate-atomic-alignment-cfg-check-on-stable/87863/2
  // for background on how to verify compatible platforms.
  let targets_that_have_equal_alignments = &[
    "wasm32-unknown-unknown",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "x86_64-pc-windows-msvc",
    "x86_64-pc-windows-gnu",
    "x86_64-apple-darwin",
  ];
  if targets_that_have_equal_alignments.contains(&target.as_str()) {
    println!("cargo:rustc-cfg=stable_target_has_atomic_equal_alignment");
  }
}
