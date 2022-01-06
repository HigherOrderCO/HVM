#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

//use cranelift::prelude::*;
use core::mem;
use cranelift::codegen::entity::EntityRef;
use cranelift::codegen::ir::types::*;
use cranelift::codegen::ir::{AbiParam, ExternalName, Function, InstBuilder, Signature};
use cranelift::codegen::isa::CallConv;
use cranelift::codegen::settings;
use cranelift::codegen::verifier::verify_function;
use cranelift::codegen::{Context};
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataContext, Linkage, Module};
use cranelift::prelude::Configurable;

fn main() -> Result<(), String> {
  let mut jit = jit_init();

  // JIT-compiles an `λx.(x + 42)` function
  let func = make_adder(&mut jit, "fn0", 42)?;
  println!("uau: {}", func(7)); // output: 49

  // JIT-compiles an `λx.(x + 10)` function
  let func = make_adder(&mut jit, "fn1", 10)?;
  println!("uau: {}", func(7)); // output: 17

  return Ok(());
}

// Static objects used by the JIT compiler
pub struct JIT {
  builder_ctx: FunctionBuilderContext,
  codegen_ctx: Context,
  data_ctx: DataContext,
  module: JITModule,
}

// Builds the static objects used by the JIT compiler
fn jit_init() -> JIT {
  let mut flag_builder = settings::builder();
  flag_builder.set("use_colocated_libcalls", "false").unwrap();
  flag_builder.set("is_pic", "false").unwrap();
  let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| { panic!("host machine is not supported: {}", msg); });
  let isa = isa_builder.finish(settings::Flags::new(flag_builder));
  let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
  let module = JITModule::new(builder);
  let builder_ctx = FunctionBuilderContext::new();
  let codegen_ctx = module.make_context();
  let data_ctx = DataContext::new();
  return JIT {
    module,
    builder_ctx,
    codegen_ctx,
    data_ctx
  }
}

// Returns a JIT-compiled `u64 -> u64` adder function. Uses the following IR template:
//   function u0:0(u64) -> u64 apple_aarch64 {
//   block0(v0: u64):
//     v1 = iconst.u64 _AMOUNT_
//     v2 = iadd v0, v1
//     return v2
//   }
fn make_adder(jit: &mut JIT, name: &str, amount : i64) -> Result<fn(u64) -> u64, String> {
  // Signature is the function type. Here, `fn(u64) -> u64`.
  jit.codegen_ctx.func.signature.params.push(AbiParam::new(I64));
  jit.codegen_ctx.func.signature.returns.push(AbiParam::new(I64));

  // TODO: I don't know how is this different from FunctionBuilderContext
  let mut builder = FunctionBuilder::new(&mut jit.codegen_ctx.func, &mut jit.builder_ctx);

  // Declares 3 local vars
  // > let x : u64;
  // > let y : u64;
  // > let z : u64;
  let x = Variable::new(0);
  let y = Variable::new(1);
  let z = Variable::new(2);
  builder.declare_var(x, I64);
  builder.declare_var(y, I64);
  builder.declare_var(z, I64);

  // Creates a block
  let block0 = builder.create_block();

  // Adds the function's params to the block
  builder.append_block_params_for_function_params(block0);

  // Focuses the block
  builder.switch_to_block(block0);

  // Signals that nothing after that branches to the block
  builder.seal_block(block0);

  // > let x = arg_0;
  let tmp = builder.block_params(block0)[0];
  builder.def_var(x, tmp);

  // > let y = 42;
  let tmp = builder.ins().iconst(I64, amount); // todo: how to use u64?
  builder.def_var(y, tmp);

  // > let z = x + y;
  let arg1 = builder.use_var(x);
  let arg2 = builder.use_var(y);
  let tmp = builder.ins().iadd(arg1, arg2);
  builder.def_var(z, tmp);

  // return z
  builder.ins().return_(&[tmp]);

  builder.finalize();

  // TODO: not sure what this does yet
  let flags = settings::Flags::new(settings::builder());
  let res = verify_function(&jit.codegen_ctx.func, &flags);
  println!("{}", jit.codegen_ctx.func.display());
  if let Err(errors) = res {
      panic!("{}", errors);
  }

  // Declares the function
  let func = jit.module.declare_function(name,
    Linkage::Export,
    &jit.codegen_ctx.func.signature
  ).map_err(|e| e.to_string())?;

  // Defines the function
  jit.module.define_function(func,
    &mut jit.codegen_ctx,
    &mut cranelift::codegen::binemit::NullTrapSink {},
    &mut cranelift::codegen::binemit::NullStackMapSink {}
  ).map_err(|e| e.to_string())?;

  // Cleanup ?
  jit.module.clear_context(&mut jit.codegen_ctx);
  jit.module.finalize_definitions();

  // This is the assembly!
  let code = jit.module.get_finalized_function(func);

  unsafe {
    let func = mem::transmute::<_, fn(u64) -> u64>(code);
    return Ok(func);
  }
}
