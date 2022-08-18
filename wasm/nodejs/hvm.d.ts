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
}
