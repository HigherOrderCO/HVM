import * as wasm from './hvm_bg.wasm';

const lTextDecoder = typeof TextDecoder === 'undefined' ? (0, module.require)('util').TextDecoder : TextDecoder;

let cachedTextDecoder = new lTextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

let cachedUint8Memory0 = new Uint8Array();

function getUint8Memory0() {
    if (cachedUint8Memory0.byteLength === 0) {
        cachedUint8Memory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8Memory0;
}

function getStringFromWasm0(ptr, len) {
    return cachedTextDecoder.decode(getUint8Memory0().subarray(ptr, ptr + len));
}

const heap = new Array(32).fill(undefined);

heap.push(undefined, null, true, false);

let heap_next = heap.length;

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

let cachedInt32Memory0 = new Int32Array();

function getInt32Memory0() {
    if (cachedInt32Memory0.byteLength === 0) {
        cachedInt32Memory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32Memory0;
}

const u32CvtShim = new Uint32Array(2);

const uint64CvtShim = new BigUint64Array(u32CvtShim.buffer);

let WASM_VECTOR_LEN = 0;

const lTextEncoder = typeof TextEncoder === 'undefined' ? (0, module.require)('util').TextEncoder : TextEncoder;

let cachedTextEncoder = new lTextEncoder('utf-8');

const encodeString = (typeof cachedTextEncoder.encodeInto === 'function'
    ? function (arg, view) {
    return cachedTextEncoder.encodeInto(arg, view);
}
    : function (arg, view) {
    const buf = cachedTextEncoder.encode(arg);
    view.set(buf);
    return {
        read: arg.length,
        written: buf.length
    };
});

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length);
        getUint8Memory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len);

    const mem = getUint8Memory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3);
        const view = getUint8Memory0().subarray(ptr + offset, ptr + len);
        const ret = encodeString(arg, view);

        offset += ret.written;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function getObject(idx) { return heap[idx]; }

function dropObject(idx) {
    if (idx < 36) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}
/**
*/
export class Reduced {

    __destroy_into_raw() {
        const ptr = this.ptr;
        this.ptr = 0;

        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_reduced_free(ptr);
    }
    /**
    * @returns {string}
    */
    get_norm() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.reduced_get_norm(retptr, this.ptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_free(r0, r1);
        }
    }
    /**
    * @returns {bigint}
    */
    get_cost() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.reduced_get_cost(retptr, this.ptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    get_size() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.reduced_get_size(retptr, this.ptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    get_time() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.reduced_get_time(retptr, this.ptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
/**
*/
export class Runtime {

    static __wrap(ptr) {
        const obj = Object.create(Runtime.prototype);
        obj.ptr = ptr;

        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.ptr;
        this.ptr = 0;

        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_runtime_free(ptr);
    }
    /**
    * Creates a new, empty runtime
    * @param {number} size
    * @returns {Runtime}
    */
    static new(size) {
        const ret = wasm.runtime_new(size);
        return Runtime.__wrap(ret);
    }
    /**
    * Creates a runtime from source code, given a max number of nodes
    * @param {string} code
    * @param {number} size
    * @returns {Runtime}
    */
    static from_code_with_size(code, size) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(code, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            wasm.runtime_from_code_with_size(retptr, ptr0, len0, size);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            var r2 = getInt32Memory0()[retptr / 4 + 2];
            if (r2) {
                throw takeObject(r1);
            }
            return Runtime.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {string} code
    * @returns {Runtime}
    */
    static from_code(code) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(code, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            wasm.runtime_from_code(retptr, ptr0, len0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            var r2 = getInt32Memory0()[retptr / 4 + 2];
            if (r2) {
                throw takeObject(r1);
            }
            return Runtime.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * Extends a runtime with new definitions
    * @param {string} _code
    */
    define(_code) {
        const ptr0 = passStringToWasm0(_code, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.runtime_define(this.ptr, ptr0, len0);
    }
    /**
    * Allocates a new term, returns its location
    * @param {string} code
    * @returns {bigint}
    */
    alloc_code(code) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(code, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            wasm.runtime_alloc_code(retptr, this.ptr, ptr0, len0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            var r2 = getInt32Memory0()[retptr / 4 + 2];
            var r3 = getInt32Memory0()[retptr / 4 + 3];
            if (r3) {
                throw takeObject(r2);
            }
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * Given a location, returns the pointer stored on it
    * @param {bigint} host
    * @returns {bigint}
    */
    at(host) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = host;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_at(retptr, this.ptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * Given a location, evaluates a term to head normal form
    * @param {bigint} host
    */
    reduce(host) {
        uint64CvtShim[0] = host;
        const low0 = u32CvtShim[0];
        const high0 = u32CvtShim[1];
        wasm.runtime_reduce(this.ptr, low0, high0);
    }
    /**
    * Given a location, evaluates a term to full normal form
    * @param {bigint} host
    */
    normalize(host) {
        uint64CvtShim[0] = host;
        const low0 = u32CvtShim[0];
        const high0 = u32CvtShim[1];
        wasm.runtime_normalize(this.ptr, low0, high0);
    }
    /**
    * Evaluates a code, allocs and evaluates to full normal form. Returns its location.
    * @param {string} code
    * @returns {bigint}
    */
    normalize_code(code) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(code, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            wasm.runtime_eval_to_loc(retptr, this.ptr, ptr0, len0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * Evaluates a code to normal form. Returns its location.
    * @param {string} code
    * @returns {bigint}
    */
    eval_to_loc(code) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(code, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            wasm.runtime_eval_to_loc(retptr, this.ptr, ptr0, len0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * Evaluates a code to normal form.
    * @param {string} code
    * @returns {string}
    */
    eval(code) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(code, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            wasm.runtime_eval(retptr, this.ptr, ptr0, len0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_free(r0, r1);
        }
    }
    /**
    * Given a location, recovers the lambda Term stored on it, as code
    * @param {bigint} host
    * @returns {string}
    */
    show(host) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = host;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_show(retptr, this.ptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_free(r0, r1);
        }
    }
    /**
    * Given a location, recovers the linear Term stored on it, as code
    * @param {bigint} host
    * @returns {string}
    */
    show_linear(host) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = host;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_show_linear(retptr, this.ptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_free(r0, r1);
        }
    }
    /**
    * Return the total number of graph rewrites computed
    * @returns {bigint}
    */
    get_rewrites() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.reduced_get_time(retptr, this.ptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * Returns the name of a given id
    * @param {bigint} id
    * @returns {string}
    */
    get_name(id) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = id;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_get_name(retptr, this.ptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_free(r0, r1);
        }
    }
    /**
    * Returns the arity of a given id
    * @param {bigint} id
    * @returns {bigint}
    */
    get_arity(id) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = id;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_get_arity(retptr, this.ptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * Returns the name of a given id
    * @param {string} name
    * @returns {bigint}
    */
    get_id(name) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            wasm.runtime_get_id(retptr, this.ptr, ptr0, len0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static DP0() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_ADD(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static DP1() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_DP1(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static VAR() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_MUL(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static ARG() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_ARG(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static ERA() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_ERA(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static LAM() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_AND(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static APP() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_APP(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static SUP() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_SUP(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static CTR() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_CTR(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static FUN() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_FUN(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static OP2() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_LTN(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static NUM() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_LTE(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static ADD() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_ADD(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static SUB() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_DP1(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static MUL() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_MUL(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static DIV() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_ARG(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static MOD() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_ERA(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static AND() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_AND(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static OR() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_APP(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static XOR() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_SUP(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static SHL() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_CTR(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static SHR() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_FUN(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static LTN() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_LTN(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static LTE() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_LTE(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static EQL() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_EQL(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static GTE() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_GTE(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static GTN() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_GTN(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static NEQ() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_NEQ(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {number}
    */
    static CELLS_PER_KB() {
        const ret = wasm.runtime_CELLS_PER_KB();
        return ret >>> 0;
    }
    /**
    * @returns {number}
    */
    static CELLS_PER_MB() {
        const ret = wasm.runtime_CELLS_PER_MB();
        return ret >>> 0;
    }
    /**
    * @returns {number}
    */
    static CELLS_PER_GB() {
        const ret = wasm.runtime_CELLS_PER_GB();
        return ret >>> 0;
    }
    /**
    * @param {bigint} lnk
    * @returns {bigint}
    */
    static get_tag(lnk) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = lnk;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_get_tag(retptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} lnk
    * @returns {bigint}
    */
    static get_ext(lnk) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = lnk;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_get_ext(retptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} lnk
    * @returns {bigint}
    */
    static get_val(lnk) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = lnk;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_get_val(retptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} lnk
    * @returns {bigint}
    */
    static get_num(lnk) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = lnk;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_get_num(retptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} lnk
    * @param {bigint} arg
    * @returns {bigint}
    */
    static get_loc(lnk, arg) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = lnk;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            uint64CvtShim[0] = arg;
            const low1 = u32CvtShim[0];
            const high1 = u32CvtShim[1];
            wasm.runtime_get_loc(retptr, low0, high0, low1, high1);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n2 = uint64CvtShim[0];
            return n2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} pos
    * @returns {bigint}
    */
    static Var(pos) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = pos;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_Var(retptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} col
    * @param {bigint} pos
    * @returns {bigint}
    */
    static Dp0(col, pos) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = col;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            uint64CvtShim[0] = pos;
            const low1 = u32CvtShim[0];
            const high1 = u32CvtShim[1];
            wasm.runtime_Dp0(retptr, low0, high0, low1, high1);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n2 = uint64CvtShim[0];
            return n2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} col
    * @param {bigint} pos
    * @returns {bigint}
    */
    static Dp1(col, pos) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = col;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            uint64CvtShim[0] = pos;
            const low1 = u32CvtShim[0];
            const high1 = u32CvtShim[1];
            wasm.runtime_Dp1(retptr, low0, high0, low1, high1);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n2 = uint64CvtShim[0];
            return n2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} pos
    * @returns {bigint}
    */
    static Arg(pos) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = pos;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_Arg(retptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @returns {bigint}
    */
    static Era() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.runtime_Era(retptr);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n0 = uint64CvtShim[0];
            return n0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} pos
    * @returns {bigint}
    */
    static Lam(pos) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = pos;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_Lam(retptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} pos
    * @returns {bigint}
    */
    static App(pos) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = pos;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_App(retptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} col
    * @param {bigint} pos
    * @returns {bigint}
    */
    static Par(col, pos) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = col;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            uint64CvtShim[0] = pos;
            const low1 = u32CvtShim[0];
            const high1 = u32CvtShim[1];
            wasm.runtime_Par(retptr, low0, high0, low1, high1);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n2 = uint64CvtShim[0];
            return n2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} ope
    * @param {bigint} pos
    * @returns {bigint}
    */
    static Op2(ope, pos) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = ope;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            uint64CvtShim[0] = pos;
            const low1 = u32CvtShim[0];
            const high1 = u32CvtShim[1];
            wasm.runtime_Op2(retptr, low0, high0, low1, high1);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n2 = uint64CvtShim[0];
            return n2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} val
    * @returns {bigint}
    */
    static Num(val) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = val;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_Num(retptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} fun
    * @param {bigint} pos
    * @returns {bigint}
    */
    static Ctr(fun, pos) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = fun;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            uint64CvtShim[0] = pos;
            const low1 = u32CvtShim[0];
            const high1 = u32CvtShim[1];
            wasm.runtime_Ctr(retptr, low0, high0, low1, high1);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n2 = uint64CvtShim[0];
            return n2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} fun
    * @param {bigint} pos
    * @returns {bigint}
    */
    static Fun(fun, pos) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = fun;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            uint64CvtShim[0] = pos;
            const low1 = u32CvtShim[0];
            const high1 = u32CvtShim[1];
            wasm.runtime_Fun(retptr, low0, high0, low1, high1);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n2 = uint64CvtShim[0];
            return n2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} loc
    * @param {bigint} lnk
    * @returns {bigint}
    */
    link(loc, lnk) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = loc;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            uint64CvtShim[0] = lnk;
            const low1 = u32CvtShim[0];
            const high1 = u32CvtShim[1];
            wasm.runtime_link(retptr, this.ptr, low0, high0, low1, high1);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n2 = uint64CvtShim[0];
            return n2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} size
    * @returns {bigint}
    */
    alloc(size) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            uint64CvtShim[0] = size;
            const low0 = u32CvtShim[0];
            const high0 = u32CvtShim[1];
            wasm.runtime_alloc(retptr, this.ptr, low0, high0);
            var r0 = getInt32Memory0()[retptr / 4 + 0];
            var r1 = getInt32Memory0()[retptr / 4 + 1];
            u32CvtShim[0] = r0;
            u32CvtShim[1] = r1;
            const n1 = uint64CvtShim[0];
            return n1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
    * @param {bigint} loc
    * @param {bigint} size
    */
    clear(loc, size) {
        uint64CvtShim[0] = loc;
        const low0 = u32CvtShim[0];
        const high0 = u32CvtShim[1];
        uint64CvtShim[0] = size;
        const low1 = u32CvtShim[0];
        const high1 = u32CvtShim[1];
        wasm.runtime_clear(this.ptr, low0, high0, low1, high1);
    }
    /**
    * @param {bigint} term
    */
    collect(term) {
        uint64CvtShim[0] = term;
        const low0 = u32CvtShim[0];
        const high0 = u32CvtShim[1];
        wasm.runtime_collect(this.ptr, low0, high0);
    }
}

export function __wbindgen_string_new(arg0, arg1) {
    const ret = getStringFromWasm0(arg0, arg1);
    return addHeapObject(ret);
};

export function __wbindgen_throw(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
};

