use std::process::Command;

fn main() {
  let cores = num_cpus::get();
  let tpcl2 = (cores as f64).log2().floor() as u32;

  println!("cargo:rerun-if-changed=src/run.c");
  println!("cargo:rerun-if-changed=src/hvm.c");
  println!("cargo:rerun-if-changed=src/run.cu");
  println!("cargo:rerun-if-changed=src/hvm.cu");
  println!("cargo:rerun-if-changed=src/get_shared_mem.cu");
  println!("cargo:rerun-if-changed=src/shared_mem_config.h");
  println!("cargo:rustc-link-arg=-rdynamic");

  match cc::Build::new()
      .file("src/run.c")
      .opt_level(3)
      .warnings(false)
      .define("TPC_L2", &*tpcl2.to_string())
      .define("IO", None)
      .try_compile("hvm-c") {
    Ok(_) => println!("cargo:rustc-cfg=feature=\"c\""),
    Err(e) => {
      println!("cargo:warning=\x1b[1m\x1b[31mWARNING: Failed to compile/run.c:\x1b[0m {}", e);
      println!("cargo:warning=Ignoring/run.c and proceeding with build. \x1b[1mThe C runtime will not be available.\x1b[0m");
    }
  }

  // Builds hvm.cu
  if Command::new("nvcc").arg("--version").stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().is_ok() {
    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
      println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
      println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }

    // Compile get_shared_mem.cu
    if let Ok(output) = Command::new("nvcc")
      .args(&["src/get_shared_mem.cu", "-o", "get_shared_mem"])
      .output()
      .and_then(|_| Command::new("./get_shared_mem").output()) {
        if output.status.success() {
          let shared_mem_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
          std::fs::write("src/shared_mem_config.h", format!("#define HVM_SHARED_MEM {}", shared_mem_str))
            .expect("Failed to write shared_mem_config.h");
          println!("cargo:warning=Shared memory size: {}", shared_mem_str);
        } else {
          println!("cargo:warning=\x1b[1m\x1b[31mWARNING: Failed to get shared memory size. Using default value.\x1b[0m");
        }
    } else {
      println!("cargo:warning=\x1b[1m\x1b[31mWARNING: Failed to compile or run get_shared_mem.cu. Using default shared memory value.\x1b[0m");
    }

    // Clean up temporary executable
    let _ = std::fs::remove_file("get_shared_mem");

    cc::Build::new()
      .cuda(true)
      .file("src/run.cu")
      .define("IO", None)
      .flag("-diag-suppress=177") // variable was declared but never referenced
      .flag("-diag-suppress=550") // variable was set but never used
      .flag("-diag-suppress=20039") // a __host__ function redeclared with __device__, hence treated as a __host__ __device__ function
      .compile("hvm-cu");

    println!("cargo:rustc-cfg=feature=\"cuda\"");
  }
  else {
    println!("cargo:warning=\x1b[1m\x1b[31mWARNING: CUDA compiler not found.\x1b[0m \x1b[1mHVM will not be able to run on GPU.\x1b[0m");
  }
}
