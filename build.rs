fn main() {
  // Builds hvm.c
  cc::Build::new()
    .file("src/hvm.c")
    .opt_level(3)
    .warnings(false)
    .compile("hvm-c");
  println!("cargo:rerun-if-changed=src/hvm.c");

  // Builds hvm.cu
  if std::process::Command::new("nvcc").arg("--version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().is_ok() {
    cc::Build::new()
      .cuda(true)
      .file("src/hvm.cu")
      .opt_level(3)
      .warnings(false)
      .compile("hvm-cu");
    println!("cargo:rustc-cfg=feature=\"cuda\"");
    println!("cargo:rerun-if-changed=src/hvm.cu");
  } else {
    println!("WARNING: CUDA compiler not found. HVM will not be able to run on GPU.");
  }
}
