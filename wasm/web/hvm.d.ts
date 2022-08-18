/* tslint:disable */
/* eslint-disable */
/**
*/
export class Reduced {
  free(): void;
/**
* @returns {string}
*/
  get_norm(): string;
/**
* @returns {bigint}
*/
  get_cost(): bigint;
/**
* @returns {bigint}
*/
  get_size(): bigint;
/**
* @returns {bigint}
*/
  get_time(): bigint;
}
/**
*/
export class Runtime {
  free(): void;
/**
* Creates a new, empty runtime
* @param {number} size
* @returns {Runtime}
*/
  static new(size: number): Runtime;
/**
* Creates a runtime from source code, given a max number of nodes
* @param {string} code
* @param {number} size
* @returns {Runtime}
*/
  static from_code_with_size(code: string, size: number): Runtime;
/**
* @param {string} code
* @returns {Runtime}
*/
  static from_code(code: string): Runtime;
/**
* Extends a runtime with new definitions
* @param {string} _code
*/
  define(_code: string): void;
/**
* Allocates a new term, returns its location
* @param {string} code
* @returns {bigint}
*/
  alloc_code(code: string): bigint;
/**
* Given a location, returns the pointer stored on it
* @param {bigint} host
* @returns {bigint}
*/
  at(host: bigint): bigint;
/**
* Given a location, evaluates a term to head normal form
* @param {bigint} host
*/
  reduce(host: bigint): void;
/**
* Given a location, evaluates a term to full normal form
* @param {bigint} host
*/
  normalize(host: bigint): void;
/**
* Evaluates a code, allocs and evaluates to full normal form. Returns its location.
* @param {string} code
* @returns {bigint}
*/
  normalize_code(code: string): bigint;
/**
* Evaluates a code to normal form. Returns its location.
* @param {string} code
* @returns {bigint}
*/
  eval_to_loc(code: string): bigint;
/**
* Evaluates a code to normal form.
* @param {string} code
* @returns {string}
*/
  eval(code: string): string;
/**
* Given a location, runs side-efefctive actions
* @param {bigint} host
*/
  run_io(host: bigint): void;
/**
* Given a location, recovers the lambda Term stored on it, as code
* @param {bigint} host
* @returns {string}
*/
  show(host: bigint): string;
/**
* Given a location, recovers the linear Term stored on it, as code
* @param {bigint} host
* @returns {string}
*/
  show_linear(host: bigint): string;
/**
* Return the total number of graph rewrites computed
* @returns {bigint}
*/
  get_rewrites(): bigint;
/**
* Returns the name of a given id
* @param {bigint} id
* @returns {string}
*/
  get_name(id: bigint): string;
/**
* Returns the arity of a given id
* @param {bigint} id
* @returns {bigint}
*/
  get_arity(id: bigint): bigint;
/**
* Returns the name of a given id
* @param {string} name
* @returns {bigint}
*/
  get_id(name: string): bigint;
/**
* @returns {bigint}
*/
  static DP0(): bigint;
/**
* @returns {bigint}
*/
  static DP1(): bigint;
/**
* @returns {bigint}
*/
  static VAR(): bigint;
/**
* @returns {bigint}
*/
  static ARG(): bigint;
/**
* @returns {bigint}
*/
  static ERA(): bigint;
/**
* @returns {bigint}
*/
  static LAM(): bigint;
/**
* @returns {bigint}
*/
  static APP(): bigint;
/**
* @returns {bigint}
*/
  static SUP(): bigint;
/**
* @returns {bigint}
*/
  static CTR(): bigint;
/**
* @returns {bigint}
*/
  static FUN(): bigint;
/**
* @returns {bigint}
*/
  static OP2(): bigint;
/**
* @returns {bigint}
*/
  static NUM(): bigint;
/**
* @returns {bigint}
*/
  static ADD(): bigint;
/**
* @returns {bigint}
*/
  static SUB(): bigint;
/**
* @returns {bigint}
*/
  static MUL(): bigint;
/**
* @returns {bigint}
*/
  static DIV(): bigint;
/**
* @returns {bigint}
*/
  static MOD(): bigint;
/**
* @returns {bigint}
*/
  static AND(): bigint;
/**
* @returns {bigint}
*/
  static OR(): bigint;
/**
* @returns {bigint}
*/
  static XOR(): bigint;
/**
* @returns {bigint}
*/
  static SHL(): bigint;
/**
* @returns {bigint}
*/
  static SHR(): bigint;
/**
* @returns {bigint}
*/
  static LTN(): bigint;
/**
* @returns {bigint}
*/
  static LTE(): bigint;
/**
* @returns {bigint}
*/
  static EQL(): bigint;
/**
* @returns {bigint}
*/
  static GTE(): bigint;
/**
* @returns {bigint}
*/
  static GTN(): bigint;
/**
* @returns {bigint}
*/
  static NEQ(): bigint;
/**
* @returns {number}
*/
  static CELLS_PER_KB(): number;
/**
* @returns {number}
*/
  static CELLS_PER_MB(): number;
/**
* @returns {number}
*/
  static CELLS_PER_GB(): number;
/**
* @param {bigint} lnk
* @returns {bigint}
*/
  static get_tag(lnk: bigint): bigint;
/**
* @param {bigint} lnk
* @returns {bigint}
*/
  static get_ext(lnk: bigint): bigint;
/**
* @param {bigint} lnk
* @returns {bigint}
*/
  static get_val(lnk: bigint): bigint;
/**
* @param {bigint} lnk
* @returns {bigint}
*/
  static get_num(lnk: bigint): bigint;
/**
* @param {bigint} lnk
* @param {bigint} arg
* @returns {bigint}
*/
  static get_loc(lnk: bigint, arg: bigint): bigint;
/**
* @param {bigint} pos
* @returns {bigint}
*/
  static Var(pos: bigint): bigint;
/**
* @param {bigint} col
* @param {bigint} pos
* @returns {bigint}
*/
  static Dp0(col: bigint, pos: bigint): bigint;
/**
* @param {bigint} col
* @param {bigint} pos
* @returns {bigint}
*/
  static Dp1(col: bigint, pos: bigint): bigint;
/**
* @param {bigint} pos
* @returns {bigint}
*/
  static Arg(pos: bigint): bigint;
/**
* @returns {bigint}
*/
  static Era(): bigint;
/**
* @param {bigint} pos
* @returns {bigint}
*/
  static Lam(pos: bigint): bigint;
/**
* @param {bigint} pos
* @returns {bigint}
*/
  static App(pos: bigint): bigint;
/**
* @param {bigint} col
* @param {bigint} pos
* @returns {bigint}
*/
  static Par(col: bigint, pos: bigint): bigint;
/**
* @param {bigint} ope
* @param {bigint} pos
* @returns {bigint}
*/
  static Op2(ope: bigint, pos: bigint): bigint;
/**
* @param {bigint} val
* @returns {bigint}
*/
  static Num(val: bigint): bigint;
/**
* @param {bigint} fun
* @param {bigint} pos
* @returns {bigint}
*/
  static Ctr(fun: bigint, pos: bigint): bigint;
/**
* @param {bigint} fun
* @param {bigint} pos
* @returns {bigint}
*/
  static Fun(fun: bigint, pos: bigint): bigint;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_runtime_free: (a: number) => void;
  readonly __wbg_reduced_free: (a: number) => void;
  readonly reduced_get_norm: (a: number, b: number) => void;
  readonly reduced_get_cost: (a: number, b: number) => void;
  readonly reduced_get_size: (a: number, b: number) => void;
  readonly reduced_get_time: (a: number, b: number) => void;
  readonly runtime_new: (a: number) => number;
  readonly runtime_from_code_with_size: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_from_code: (a: number, b: number, c: number) => void;
  readonly runtime_define: (a: number, b: number, c: number) => void;
  readonly runtime_alloc_code: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_at: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_reduce: (a: number, b: number, c: number) => void;
  readonly runtime_normalize: (a: number, b: number, c: number) => void;
  readonly runtime_eval_to_loc: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_eval: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_run_io: (a: number, b: number, c: number) => void;
  readonly runtime_show: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_show_linear: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_get_name: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_get_arity: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_get_id: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_DP1: (a: number) => void;
  readonly runtime_ARG: (a: number) => void;
  readonly runtime_ERA: (a: number) => void;
  readonly runtime_APP: (a: number) => void;
  readonly runtime_SUP: (a: number) => void;
  readonly runtime_CTR: (a: number) => void;
  readonly runtime_FUN: (a: number) => void;
  readonly runtime_ADD: (a: number) => void;
  readonly runtime_MUL: (a: number) => void;
  readonly runtime_AND: (a: number) => void;
  readonly runtime_LTN: (a: number) => void;
  readonly runtime_LTE: (a: number) => void;
  readonly runtime_EQL: (a: number) => void;
  readonly runtime_GTE: (a: number) => void;
  readonly runtime_GTN: (a: number) => void;
  readonly runtime_NEQ: (a: number) => void;
  readonly runtime_get_tag: (a: number, b: number, c: number) => void;
  readonly runtime_get_ext: (a: number, b: number, c: number) => void;
  readonly runtime_get_val: (a: number, b: number, c: number) => void;
  readonly runtime_get_num: (a: number, b: number, c: number) => void;
  readonly runtime_get_loc: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly runtime_Var: (a: number, b: number, c: number) => void;
  readonly runtime_Dp0: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly runtime_Dp1: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly runtime_Arg: (a: number, b: number, c: number) => void;
  readonly runtime_Era: (a: number) => void;
  readonly runtime_Lam: (a: number, b: number, c: number) => void;
  readonly runtime_App: (a: number, b: number, c: number) => void;
  readonly runtime_Par: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly runtime_Op2: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly runtime_Num: (a: number, b: number, c: number) => void;
  readonly runtime_Ctr: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly runtime_Fun: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly runtime_normalize_code: (a: number, b: number, c: number, d: number) => void;
  readonly runtime_DP0: (a: number) => void;
  readonly runtime_SUB: (a: number) => void;
  readonly runtime_VAR: (a: number) => void;
  readonly runtime_DIV: (a: number) => void;
  readonly runtime_MOD: (a: number) => void;
  readonly runtime_LAM: (a: number) => void;
  readonly runtime_OR: (a: number) => void;
  readonly runtime_XOR: (a: number) => void;
  readonly runtime_SHL: (a: number) => void;
  readonly runtime_SHR: (a: number) => void;
  readonly runtime_OP2: (a: number) => void;
  readonly runtime_NUM: (a: number) => void;
  readonly runtime_CELLS_PER_KB: () => number;
  readonly runtime_CELLS_PER_MB: () => number;
  readonly runtime_CELLS_PER_GB: () => number;
  readonly runtime_get_rewrites: (a: number, b: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number) => void;
  readonly __wbindgen_malloc: (a: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number) => number;
}

/**
* Synchronously compiles the given `bytes` and instantiates the WebAssembly module.
*
* @param {BufferSource} bytes
*
* @returns {InitOutput}
*/
export function initSync(bytes: BufferSource): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
