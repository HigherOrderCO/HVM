use core::panic;
use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;
use std::{mem, thread};

use crate::ast::Net;

// const fn hacks to read the TPC environment until
// https://github.com/rust-lang/rust/pull/124941 is merged
const fn get_tpc() -> usize {
    let string = env!("TPC");
    let mut res: usize = 0;
    let mut bytes = string.as_bytes();
    while let [byte, rest @ ..] = bytes {
        bytes = rest;
        if let b'0'..=b'9' = byte {
            res *= 10;
            res += (*byte - b'0') as usize;
        } else {
            panic!("not a number")
        }
    }
    res
}
// Threads per CPU
pub const TPC: usize = get_tpc();

// Runtime
// =======

// Types
pub type Tag = u8; // Tag  ::= 3-bit (rounded up to u8)
pub type Lab = u32; // Lab  ::= 29-bit (rounded up to u32)
pub type Val = u32; // Val  ::= 29-bit (rounded up to u32)
pub type Rule = u8; // Rule ::= 8-bit (fits a u8)

// Port
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Hash)]
pub struct Port(pub Val);

// Pair
#[derive(Copy, Clone)]
pub struct Pair(pub u64);

// Atomics
pub type AVal = AtomicU32;
pub struct APort(pub AVal);
pub struct APair(pub AtomicU64);

// Number
pub struct Numb(pub Val);

// Tags
pub const VAR: Tag = 0x0; // variable
pub const REF: Tag = 0x1; // reference
pub const ERA: Tag = 0x2; // eraser
pub const NUM: Tag = 0x3; // number
pub const CON: Tag = 0x4; // constructor
pub const DUP: Tag = 0x5; // duplicator
pub const OPR: Tag = 0x6; // operator
pub const SWI: Tag = 0x7; // switch

// Rules
pub const LINK: Rule = 0x0;
pub const CALL: Rule = 0x1;
pub const VOID: Rule = 0x2;
pub const ERAS: Rule = 0x3;
pub const ANNI: Rule = 0x4;
pub const COMM: Rule = 0x5;
pub const OPER: Rule = 0x6;
pub const SWIT: Rule = 0x7;

// Numbs
pub const TY_SYM: Tag = 0x00;
pub const TY_U24: Tag = 0x01;
pub const TY_I24: Tag = 0x02;
pub const TY_F24: Tag = 0x03;
pub const OP_ADD: Tag = 0x04;
pub const OP_SUB: Tag = 0x05;
pub const FP_SUB: Tag = 0x06;
pub const OP_MUL: Tag = 0x07;
pub const OP_DIV: Tag = 0x08;
pub const FP_DIV: Tag = 0x09;
pub const OP_REM: Tag = 0x0A;
pub const FP_REM: Tag = 0x0B;
pub const OP_EQ: Tag = 0x0C;
pub const OP_NEQ: Tag = 0x0D;
pub const OP_LT: Tag = 0x0E;
pub const OP_GT: Tag = 0x0F;
pub const OP_AND: Tag = 0x10;
pub const OP_OR: Tag = 0x11;
pub const OP_XOR: Tag = 0x12;
pub const OP_SHL: Tag = 0x13;
pub const FP_SHL: Tag = 0x14;
pub const OP_SHR: Tag = 0x15;
pub const FP_SHR: Tag = 0x16;

// Constants
pub const FREE: Port = Port(0x0);
pub const ROOT: Port = Port(0xFFFFFFF8);
pub const NONE: Port = Port(0xFFFFFFFF);

pub const HLEN: usize = 1 << 16;
pub const RLEN: usize = 1 << 24;
pub const G_NODE_LEN: usize = 1 << 29; // max 536m nodes
pub const G_VARS_LEN: usize = 1 << 29; // max 536m vars
pub const G_RBAG_LEN: usize = TPC * RLEN;

// IO Magic Number
pub const IO_MAGIC_0: u32 = 0xD0CA11;
pub const IO_MAGIC_1: u32 = 0xFF1FF1;

// IO Tags
pub const IO_DONE: u32 = 0;
pub const IO_CALL: u32 = 1;

// RBag
pub struct RBag {
    pub base: usize,
    pub lo: Vec<Pair>,
    pub hi: Vec<Pair>,
}

// Global Net
pub struct GNet<'a> {
    pub nlen: usize,           // length of the node buffer
    pub vlen: usize,           // length of the vars buffer
    pub node: &'a mut [APair], // node buffer
    pub vars: &'a mut [APort], // vars buffer
    pub rbag: &'a mut [APair], // rbag buffer
    pub itrs: AtomicU64,       // interaction count
    pub idle: AtomicU64,       // idle thread count
}

// Thread Memory
pub struct TMem {
    pub tid: usize,           // thread id
    pub itrs: u32,            // interaction count
    pub nput: usize,          // next node allocation index
    pub vput: usize,          // next vars allocation index
    pub rput: usize,          // next rbag allocation index
    pub nloc: [usize; 0xFFF], // allocated node locations
    pub vloc: [usize; 0xFFF], // allocated vars locations
    pub rbag: RBag,           // local redexes
}

// Top-Level Definition
pub struct Def {
    pub name: String,    // def name
    pub safe: bool,      // has no dups
    pub root: Port,      // root port
    pub rbag: Vec<Pair>, // def redex bag
    pub node: Vec<Pair>, // def node buffer
    pub vars: usize,     // def vars count
}

// Book of Definitions
pub struct Book {
    pub defs: Vec<Def>,
}

impl Port {
    pub fn new(tag: Tag, val: Val) -> Self {
        Port((val << 3) | tag as Val)
    }

    pub fn get_tag(&self) -> Tag {
        (self.0 & 7) as Tag
    }

    pub fn get_val(&self) -> Val {
        self.0 >> 3
    }

    pub fn is_nod(&self) -> bool {
        self.get_tag() >= CON
    }

    pub fn is_var(&self) -> bool {
        self.get_tag() == VAR
    }

    pub fn get_rule(a: Port, b: Port) -> Rule {
        const TABLE: [[Rule; 8]; 8] = [
            //VAR   REF   ERA   NUM   CON   DUP   OPR   SWI
            [LINK, LINK, LINK, LINK, LINK, LINK, LINK, LINK], // VAR
            [LINK, VOID, VOID, VOID, CALL, CALL, CALL, CALL], // REF
            [LINK, VOID, VOID, VOID, ERAS, ERAS, ERAS, ERAS], // ERA
            [LINK, VOID, VOID, VOID, ERAS, ERAS, OPER, SWIT], // NUM
            [LINK, CALL, ERAS, ERAS, ANNI, COMM, COMM, COMM], // CON
            [LINK, CALL, ERAS, ERAS, COMM, ANNI, COMM, COMM], // DUP
            [LINK, CALL, ERAS, OPER, COMM, COMM, ANNI, COMM], // OPR
            [LINK, CALL, ERAS, SWIT, COMM, COMM, COMM, ANNI], // SWI
        ];
        return TABLE[a.get_tag() as usize][b.get_tag() as usize];
    }

    pub fn should_swap(a: Port, b: Port) -> bool {
        b.get_tag() < a.get_tag()
    }

    pub fn is_high_priority(rule: Rule) -> bool {
        (0b00011101 >> rule) & 1 != 0
    }

    pub fn adjust_port(&self, tm: &TMem) -> Port {
        let tag = self.get_tag();
        let val = self.get_val();
        if self.is_nod() {
            Port::new(tag, tm.nloc[val as usize] as u32)
        } else if self.is_var() {
            Port::new(tag, tm.vloc[val as usize] as u32)
        } else {
            Port::new(tag, val)
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl Pair {
    pub fn new(fst: Port, snd: Port) -> Self {
        Pair(((snd.0 as u64) << 32) | fst.0 as u64)
    }

    pub fn get_fst(&self) -> Port {
        Port((self.0 & 0xFFFFFFFF) as u32)
    }

    pub fn get_snd(&self) -> Port {
        Port((self.0 >> 32) as u32)
    }

    pub fn adjust_pair(&self, tm: &TMem) -> Pair {
        let p1 = self.get_fst().adjust_port(tm);
        let p2 = self.get_snd().adjust_port(tm);
        Pair::new(p1, p2)
    }

    pub fn set_par_flag(&self) -> Self {
        let p1: Port = self.get_fst();
        let p2: Port = self.get_snd();
        if p1.get_tag() == REF {
            return Pair::new(Port::new(p1.get_tag(), p1.get_val() | 0x10000000), p2);
        } else {
            return Pair::new(p1, p2);
        }
    }

    pub fn get_par_flag(&self) -> bool {
        let p1: Port = self.get_fst();
        if p1.get_tag() == REF {
            return p1.get_val() >> 28 == 1;
        } else {
            return false;
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl Numb {
    // SYM: a symbolic operator

    pub fn new_sym(val: Tag) -> Self {
        Numb((val as Val) << 5 | (TY_SYM as Val))
    }

    pub fn get_sym(&self) -> Tag {
        (self.0 >> 5) as Tag
    }

    // U24: unsigned 24-bit integer

    pub fn new_u24(val: u32) -> Self {
        Numb((val << 5) as Val | (TY_U24 as Val))
    }

    pub fn get_u24(&self) -> u32 {
        (self.0 >> 5) as u32
    }

    // I24: signed 24-bit integer

    pub fn new_i24(val: i32) -> Self {
        Numb(((val as u32) << 5) as Val | (TY_I24 as Val))
    }

    pub fn get_i24(&self) -> i32 {
        (self.0 as i32) << 3 >> 8
    }

    // F24: 24-bit float

    pub fn new_f24(val: f32) -> Self {
        let bits = val.to_bits();
        let mut shifted_bits = bits >> 8;
        let lost_bits = bits & 0xFF;
        // round ties to even
        shifted_bits +=
            u32::from(!val.is_nan()) & ((lost_bits - ((lost_bits >> 7) & !shifted_bits)) >> 7);
        // ensure NaNs don't become infinities
        shifted_bits |= u32::from(val.is_nan());
        Numb((shifted_bits << 5) as Val | (TY_F24 as Val))
    }

    pub fn get_f24(&self) -> f32 {
        f32::from_bits((self.0 << 3) & 0xFFFFFF00)
    }

    // Gets the numeric type.

    pub fn get_typ(&self) -> Tag {
        return (self.0 & 0x1F) as Tag;
    }

    // Flip flag.

    // Partial application.
    pub fn partial(a: Self, b: Self) -> Self {
        return Numb((b.0 & !0x1F) | a.get_sym() as u32);
    }

    pub fn operate(a: Self, b: Self) -> Self {
        //println!("operate {} {}", crate::ast::Numb(a.0).show(), crate::ast::Numb(b.0).show());
        let at = a.get_typ();
        let bt = b.get_typ();
        if at == TY_SYM && bt == TY_SYM {
            return Numb::new_u24(0);
        }
        if at == TY_SYM && bt != TY_SYM {
            return Numb::partial(a, b);
        }
        if at != TY_SYM && bt == TY_SYM {
            return Numb::partial(b, a);
        }
        if at >= OP_ADD && bt >= OP_ADD {
            return Numb::new_u24(0);
        }
        if at < OP_ADD && bt < OP_ADD {
            return Numb::new_u24(0);
        }
        let (op, a, ty, b) = if at >= OP_ADD {
            (at, a, bt, b)
        } else {
            (bt, b, at, a)
        };
        match ty {
            TY_U24 => {
                let av = a.get_u24();
                let bv = b.get_u24();
                match op {
                    OP_ADD => Numb::new_u24(av.wrapping_add(bv)),
                    OP_SUB => Numb::new_u24(av.wrapping_sub(bv)),
                    FP_SUB => Numb::new_u24(bv.wrapping_sub(av)),
                    OP_MUL => Numb::new_u24(av.wrapping_mul(bv)),
                    OP_DIV => Numb::new_u24(av.wrapping_div(bv)),
                    FP_DIV => Numb::new_u24(bv.wrapping_div(av)),
                    OP_REM => Numb::new_u24(av.wrapping_rem(bv)),
                    FP_REM => Numb::new_u24(bv.wrapping_rem(av)),
                    OP_EQ => Numb::new_u24((av == bv) as u32),
                    OP_NEQ => Numb::new_u24((av != bv) as u32),
                    OP_LT => Numb::new_u24((av < bv) as u32),
                    OP_GT => Numb::new_u24((av > bv) as u32),
                    OP_AND => Numb::new_u24(av & bv),
                    OP_OR => Numb::new_u24(av | bv),
                    OP_XOR => Numb::new_u24(av ^ bv),
                    OP_SHL => Numb::new_u24(av << (bv & 31)),
                    OP_SHR => Numb::new_u24(av >> (bv & 31)),
                    FP_SHL => Numb::new_u24(bv << (av & 31)),
                    FP_SHR => Numb::new_u24(bv >> (av & 31)),
                    _ => unreachable!(),
                }
            }
            TY_I24 => {
                let av = a.get_i24();
                let bv = b.get_i24();
                match op {
                    OP_ADD => Numb::new_i24(av.wrapping_add(bv)),
                    OP_SUB => Numb::new_i24(av.wrapping_sub(bv)),
                    FP_SUB => Numb::new_i24(bv.wrapping_sub(av)),
                    OP_MUL => Numb::new_i24(av.wrapping_mul(bv)),
                    OP_DIV => Numb::new_i24(av.wrapping_div(bv)),
                    FP_DIV => Numb::new_i24(bv.wrapping_div(av)),
                    OP_REM => Numb::new_i24(av.wrapping_rem(bv)),
                    FP_REM => Numb::new_i24(bv.wrapping_rem(av)),
                    OP_EQ => Numb::new_u24((av == bv) as u32),
                    OP_NEQ => Numb::new_u24((av != bv) as u32),
                    OP_LT => Numb::new_u24((av < bv) as u32),
                    OP_GT => Numb::new_u24((av > bv) as u32),
                    OP_AND => Numb::new_i24(av & bv),
                    OP_OR => Numb::new_i24(av | bv),
                    OP_XOR => Numb::new_i24(av ^ bv),
                    _ => unreachable!(),
                }
            }
            TY_F24 => {
                let av = a.get_f24();
                let bv = b.get_f24();
                match op {
                    OP_ADD => Numb::new_f24(av + bv),
                    OP_SUB => Numb::new_f24(av - bv),
                    FP_SUB => Numb::new_f24(bv - av),
                    OP_MUL => Numb::new_f24(av * bv),
                    OP_DIV => Numb::new_f24(av / bv),
                    FP_DIV => Numb::new_f24(bv / av),
                    OP_REM => Numb::new_f24(av % bv),
                    FP_REM => Numb::new_f24(bv % av),
                    OP_EQ => Numb::new_u24((av == bv) as u32),
                    OP_NEQ => Numb::new_u24((av != bv) as u32),
                    OP_LT => Numb::new_u24((av < bv) as u32),
                    OP_GT => Numb::new_u24((av > bv) as u32),
                    OP_AND => Numb::new_f24(av.atan2(bv)),
                    OP_OR => Numb::new_f24(bv.log(av)),
                    OP_XOR => Numb::new_f24(av.powf(bv)),
                    _ => unreachable!(),
                }
            }
            _ => Numb::new_u24(0),
        }
    }
}

impl<'a> GNet<'a> {
    pub fn new(nlen: usize, vlen: usize, rlen: usize) -> Self {
        let nlay = Layout::array::<APair>(nlen).unwrap();
        let vlay = Layout::array::<APort>(vlen).unwrap();
        let rlay = Layout::array::<APair>(rlen).unwrap();
        let nptr = unsafe { alloc(nlay) as *mut APair };
        let vptr = unsafe { alloc(vlay) as *mut APort };
        let rptr = unsafe { alloc(rlay) as *mut APair };
        let node = unsafe { std::slice::from_raw_parts_mut(nptr, nlen) };
        let vars = unsafe { std::slice::from_raw_parts_mut(vptr, vlen) };
        let rbag = unsafe { std::slice::from_raw_parts_mut(rptr, rlen) };
        GNet {
            nlen,
            vlen,
            node,
            vars,
            rbag,
            itrs: AtomicU64::new(0),
            idle: AtomicU64::new(0),
        }
    }

    pub fn node_create(&self, loc: usize, val: Pair) {
        self.node[loc].0.store(val.0, Ordering::Relaxed);
    }

    pub fn vars_create(&self, var: usize, val: Port) {
        self.vars[var].0.store(val.0, Ordering::Relaxed);
    }

    pub fn rbag_create(&self, idx: usize, val: Pair) {
        self.rbag[idx].0.store(val.0, Ordering::Relaxed);
    }

    pub fn node_load(&self, loc: usize) -> Pair {
        Pair(self.node[loc].0.load(Ordering::Relaxed))
    }

    pub fn vars_load(&self, var: usize) -> Port {
        Port(self.vars[var].0.load(Ordering::Relaxed) as u32)
    }

    pub fn node_store(&self, loc: usize, val: Pair) {
        self.node[loc].0.store(val.0, Ordering::Relaxed);
    }

    pub fn vars_store(&self, var: usize, val: Port) {
        self.vars[var].0.store(val.0, Ordering::Relaxed);
    }

    pub fn node_exchange(&self, loc: usize, val: Pair) -> Pair {
        Pair(self.node[loc].0.swap(val.0, Ordering::Relaxed))
    }

    pub fn vars_exchange(&self, var: usize, val: Port) -> Port {
        Port(self.vars[var].0.swap(val.0, Ordering::Relaxed) as u32)
    }

    pub fn rbag_exchange(&self, idx: usize, val: Pair) -> Pair {
        Pair(self.rbag[idx].0.swap(val.0, Ordering::Relaxed))
    }

    pub fn node_take(&self, loc: usize) -> Pair {
        self.node_exchange(loc, Pair(0))
    }

    pub fn vars_take(&self, var: usize) -> Port {
        self.vars_exchange(var, Port(0))
    }

    pub fn enter(&self, mut var: Port) -> Port {
        // While `B` is VAR: extend it (as an optimization)
        while var.get_tag() == VAR {
            // Takes the current `B` substitution as `B'`
            let val = self.vars_exchange(var.get_val() as usize, NONE);
            // If there was no `B'`, stop, as there is no extension
            if val == NONE || val == FREE {
                break;
            }
            // Otherwise, delete `B` (we own both) and continue as `A ~> B'`
            self.vars_take(var.get_val() as usize);
            var = val;
        }
        return var;
    }

    pub fn normalize(&self, book: &Book) {
        const IDLE: u64 = (TPC - 1) as u64;
        self.idle.store(IDLE, core::sync::atomic::Ordering::Relaxed);
        std::thread::scope(|s| {
            for mut tm in (0..TPC).map(TMem::new) {
                s.spawn(move || tm.evaluator(self, book));
            }
        });
    }

    pub fn expand(&self, book: &Book, mut port: Port) -> Port {
        port = self.peek(port);
        while port.get_tag() == REF {
            let redex = Pair::new(Port::new(REF, port.get_val()), ROOT);
            self.vars_create(ROOT.get_val() as usize, NONE);
            self.rbag_create(FREE.get_val() as usize, redex);

            self.normalize(book);
            port = self.peek(self.vars_load(ROOT.get_val() as usize));
        }
        return port;
    }

    pub fn peek(&self, mut port: Port) -> Port {
        while port.get_tag() == VAR {
            let pport = self.vars_load(port.get_val() as usize);
            if pport == NONE || pport == FREE {
                break;
            }
            port = pport;
        }
        return port;
    }
}

impl<'a> Drop for GNet<'a> {
    fn drop(&mut self) {
        let nlay = Layout::array::<APair>(self.nlen).unwrap();
        let vlay = Layout::array::<APair>(self.vlen).unwrap();
        let rlay = Layout::array::<APair>(self.rbag.len()).unwrap();
        unsafe {
            dealloc(self.node.as_mut_ptr() as *mut u8, nlay);
            dealloc(self.vars.as_mut_ptr() as *mut u8, vlay);
            dealloc(self.rbag.as_mut_ptr() as *mut u8, rlay);
        }
    }
}

impl RBag {
    pub fn new(base: usize) -> Self {
        RBag {
            base,
            lo: Vec::new(),
            hi: Vec::new(),
        }
    }

    pub fn push_redex(&mut self, net: &GNet, redex: Pair) {
        let rule = Port::get_rule(redex.get_fst(), redex.get_snd());
        if Port::is_high_priority(rule) {
            self.hi.push(redex)
        } else {
            net.rbag_exchange(self.base + self.lo.len(), redex);
            self.lo.push(redex)
        }
    }

    pub fn pop_redex(&mut self, net: &GNet) -> Pair {
        if let Some(high) = self.hi.pop() {
            high
        } else if let Some(_) = self.lo.pop() {
            // the redex that is requested back from the low priority queue may have been stolen by
            // another thread. this is why we return the exchange here rather than the value that
            // is kept in `lo`
            net.rbag_exchange(self.base + self.lo.len(), Pair(0))
        } else {
            Pair(0)
        }
    }

    pub fn len(&self) -> usize {
        self.lo.len() + self.hi.len()
    }

    pub fn has_highs(&self) -> bool {
        !self.hi.is_empty()
    }
}

impl TMem {
    // TODO: implement a TMem::new() fn
    pub fn new(tid: usize) -> Self {
        TMem {
            tid,
            itrs: 0,
            nput: 0,
            vput: 0,
            rput: 0,
            nloc: [0; 0xFFF],
            vloc: [0; 0xFFF],
            rbag: RBag::new(tid * RLEN),
        }
    }

    /// thread base. which is the starting index for the thread into the global memory
    fn tbase(&self) -> usize {
        self.tid * RLEN
    }

    pub fn node_alloc(&mut self, net: &GNet, num: usize) -> usize {
        let mut got = 0;
        while got < num {
            self.nput += 1; // index 0 reserved
            if net.node_load(self.nput % net.nlen).is_zero() {
                self.nloc[got] = self.tbase() + (self.nput % net.nlen);
                got += 1;
            }
        }
        return got;
    }

    pub fn vars_alloc(&mut self, net: &GNet, num: usize) -> usize {
        let mut got = 0;
        while got < num {
            self.vput += 1; // index 0 reserved for FREE
            if self.vput < net.vlen - 1 || net.vars_load(self.vput % net.vlen).is_zero() {
                self.vloc[got] = self.tbase() + (self.vput % net.nlen);
                got += 1;
            }
        }
        return got;
    }

    pub fn get_resources(
        &mut self,
        net: &GNet,
        _need_rbag: usize,
        need_node: usize,
        need_vars: usize,
    ) -> bool {
        let got_node = self.node_alloc(net, need_node);
        let got_vars = self.vars_alloc(net, need_vars);
        got_node >= need_node && got_vars >= need_vars
    }

    // Atomically Links `A ~ B`.
    pub fn link(&mut self, net: &GNet, mut a: Port, mut b: Port) {
        //println!("link {} ~ {}", a.show(), b.show());

        // Attempts to directionally point `A ~> B`
        loop {
            // If `A` is NODE: swap `A` and `B`, and continue
            if a.get_tag() != VAR && b.get_tag() == VAR {
                mem::swap(&mut a, &mut b);
            }

            // If `A` is NODE: create the `A ~ B` redex
            if a.get_tag() != VAR {
                self.rbag.push_redex(net, Pair::new(a, b));
                break;
            }

            // While `B` is VAR: extend it (as an optimization)
            b = net.enter(b);

            // Since `A` is VAR: point `A ~> B`.
            // Stores `A -> B`, taking the current `A` subst as `A'`
            let a_ = net.vars_exchange(a.get_val() as usize, b);
            // If there was no `A'`, stop, as we lost B's ownership
            if a_ == NONE {
                break;
            }
            // Otherwise, delete `A` (we own both) and link `A' ~ B`
            net.vars_take(a.get_val() as usize);
            a = a_;
        }
    }

    // Links `A ~ B` (as a pair).
    pub fn link_pair(&mut self, net: &GNet, ab: Pair) {
        //println!("link_pair {:016X}", ab.0);
        self.link(net, ab.get_fst(), ab.get_snd());
    }

    // The Link Interaction.
    pub fn interact_link(&mut self, net: &GNet, a: Port, b: Port) -> bool {
        // Allocates needed nodes and vars.
        if !self.get_resources(net, 1, 0, 0) {
            return false;
        }

        // Links.
        self.link_pair(net, Pair::new(a, b));

        true
    }

    // The Call Interaction.
    pub fn interact_call(&mut self, net: &GNet, a: Port, b: Port, book: &Book) -> bool {
        // eprintln!("[{:04x}] call", self.tid);

        let fid = (a.get_val() as usize) & 0xFFFFFFF;
        let def = &book.defs[fid];

        // Copy Optimization.
        if b.get_tag() == DUP {
            if def.safe {
                return self.interact_eras(net, a, b);
            } else {
                // TODO:
                // Currently, we'll not allow copying of REFs with DUPs. While this is perfectly valid on
                // IC semantics (i.e., if the user know what they're doing), this can lead to unsound
                // reductions when compiling λ-terms to HVM. So, for now, we'll just disable this feature,
                // and consider it undefined behavior. We should add a `--unsafe` flag that allows it.
                println!("ERROR: attempt to clone a non-affine global reference.\n");
                std::process::exit(0);
            }
        }

        // Allocates needed nodes and vars.
        if !self.get_resources(net, def.rbag.len() + 1, def.node.len(), def.vars as usize) {
            return false;
        }

        // Stores new vars.
        for i in 0..def.vars {
            net.vars_create(self.vloc[i], NONE);
            //println!("vars_create vars_loc[{:04X}] {:04X}", i, self.vloc[i]);
        }

        // Stores new nodes.
        for i in 0..def.node.len() {
            net.node_create(self.nloc[i], def.node[i].adjust_pair(self));
            //println!("node_create node_loc[{:04X}] {:016X}", i-1, def.node[i].0);
        }

        // Links.
        for pair in &def.rbag {
            self.link_pair(net, pair.adjust_pair(self));
        }
        self.link_pair(net, Pair::new(def.root.adjust_port(self), b));

        true
    }

    // The Void Interaction.
    pub fn interact_void(&mut self, _net: &GNet, _a: Port, _b: Port) -> bool {
        true
    }

    // The Eras Interaction.
    pub fn interact_eras(&mut self, net: &GNet, a: Port, b: Port) -> bool {
        // Allocates needed nodes and vars.
        if !self.get_resources(net, 2, 0, 0) {
            return false;
        }

        // Checks availability
        if net.node_load(b.get_val() as usize).is_zero() {
            return false;
        }

        // Loads ports.
        //eprintln!("[{:04x}] erasure on {}", self.tid, b.get_val());
        let b_ = net.node_exchange(b.get_val() as usize, Pair(0));
        let b1 = b_.get_fst();
        let b2 = b_.get_snd();

        // Links.
        self.link_pair(net, Pair::new(a, b1));
        self.link_pair(net, Pair::new(a, b2));

        true
    }

    // The Anni Interaction.
    pub fn interact_anni(&mut self, net: &GNet, a: Port, b: Port) -> bool {
        // Allocates needed nodes and vars.
        if !self.get_resources(net, 2, 0, 0) {
            return false;
        }

        // Checks availability
        if net.node_load(a.get_val() as usize).is_zero()
            || net.node_load(b.get_val() as usize).is_zero()
        {
            return false;
        }

        // Loads ports.
        // eprintln!("[{:04x}] interact anni on {}", self.tid, a.get_val());
        let a_ = net.node_take(a.get_val() as usize);
        let a1 = a_.get_fst();
        let a2 = a_.get_snd();
        // eprintln!("[{:04x}] interact anni on {}", self.tid, b.get_val());
        let b_ = net.node_take(b.get_val() as usize);
        let b1 = b_.get_fst();
        let b2 = b_.get_snd();

        // Links.
        self.link_pair(net, Pair::new(a1, b1));
        self.link_pair(net, Pair::new(a2, b2));

        return true;
    }

    // The Comm Interaction.
    pub fn interact_comm(&mut self, net: &GNet, a: Port, b: Port) -> bool {
        // Allocates needed nodes and vars.
        if !self.get_resources(net, 4, 4, 4) {
            return false;
        }

        // Checks availability
        if net.node_load(a.get_val() as usize).is_zero()
            || net.node_load(b.get_val() as usize).is_zero()
        {
            return false;
        }

        // Loads ports.
        // eprintln!("[{:04x}] interact comm on {}", self.tid, a.get_val());
        let a_ = net.node_take(a.get_val() as usize);
        let a1 = a_.get_fst();
        let a2 = a_.get_snd();
        // eprintln!("[{:04x}] interact comm on {}", self.tid, b.get_val());
        let b_ = net.node_take(b.get_val() as usize);
        let b1 = b_.get_fst();
        let b2 = b_.get_snd();

        // Stores new vars.
        net.vars_create(self.vloc[0], NONE);
        net.vars_create(self.vloc[1], NONE);
        net.vars_create(self.vloc[2], NONE);
        net.vars_create(self.vloc[3], NONE);

        // Stores new nodes.
        net.node_create(
            self.nloc[0],
            Pair::new(
                Port::new(VAR, self.vloc[0] as u32),
                Port::new(VAR, self.vloc[1] as u32),
            ),
        );
        net.node_create(
            self.nloc[1],
            Pair::new(
                Port::new(VAR, self.vloc[2] as u32),
                Port::new(VAR, self.vloc[3] as u32),
            ),
        );
        net.node_create(
            self.nloc[2],
            Pair::new(
                Port::new(VAR, self.vloc[0] as u32),
                Port::new(VAR, self.vloc[2] as u32),
            ),
        );
        net.node_create(
            self.nloc[3],
            Pair::new(
                Port::new(VAR, self.vloc[1] as u32),
                Port::new(VAR, self.vloc[3] as u32),
            ),
        );

        // Links.
        self.link_pair(
            net,
            Pair::new(Port::new(b.get_tag(), self.nloc[0] as u32), a1),
        );
        self.link_pair(
            net,
            Pair::new(Port::new(b.get_tag(), self.nloc[1] as u32), a2),
        );
        self.link_pair(
            net,
            Pair::new(Port::new(a.get_tag(), self.nloc[2] as u32), b1),
        );
        self.link_pair(
            net,
            Pair::new(Port::new(a.get_tag(), self.nloc[3] as u32), b2),
        );

        true
    }

    // The Oper Interaction.
    pub fn interact_oper(&mut self, net: &GNet, a: Port, b: Port) -> bool {
        // Allocates needed nodes and vars.
        if !self.get_resources(net, 1, 1, 0) {
            return false;
        }

        // Checks availability
        if net.node_load(b.get_val() as usize).is_zero() {
            return false;
        }

        assert_eq!(a.get_tag(), NUM);
        // Loads ports.
        let av = a.get_val();
        // eprintln!("[{:04x}] interact oper on {}", self.tid, b.get_val());
        let b_ = net.node_take(b.get_val() as usize);
        let b1 = b_.get_fst();
        let b2 = net.enter(b_.get_snd());

        // Performs operation.
        if b1.get_tag() == NUM {
            let bv = b1.get_val();
            let cv = Numb::operate(Numb(av), Numb(bv));
            self.link_pair(net, Pair::new(Port::new(NUM, cv.0), b2));
        } else {
            net.node_create(
                self.nloc[0],
                Pair::new(Port::new(a.get_tag(), Numb(a.get_val()).0), b2),
            );
            self.link_pair(net, Pair::new(b1, Port::new(OPR, self.nloc[0] as u32)));
        }

        true
    }

    // The Swit Interaction.
    pub fn interact_swit(&mut self, net: &GNet, a: Port, b: Port) -> bool {
        // Allocates needed nodes and vars.
        if !self.get_resources(net, 1, 2, 0) {
            return false;
        }

        // Checks availability
        if net.node_load(b.get_val() as usize).is_zero() {
            return false;
        }

        // Loads ports.
        let av = Numb(a.get_val()).get_u24();
        // eprintln!("[{:04x}] interact swit on {}", self.tid, b.get_val());
        let b_ = net.node_take(b.get_val() as usize);
        let b1 = b_.get_fst();
        let b2 = b_.get_snd();

        // Stores new nodes.
        if av == 0 {
            net.node_create(self.nloc[0], Pair::new(b2, Port::new(ERA, 0)));
            self.link_pair(net, Pair::new(Port::new(CON, self.nloc[0] as u32), b1));
        } else {
            net.node_create(
                self.nloc[0],
                Pair::new(Port::new(ERA, 0), Port::new(CON, self.nloc[1] as u32)),
            );
            net.node_create(
                self.nloc[1],
                Pair::new(Port::new(NUM, Numb::new_u24(av - 1).0), b2),
            );
            self.link_pair(net, Pair::new(Port::new(CON, self.nloc[0] as u32), b1));
        }

        true
    }

    // Pops a local redex and performs a single interaction.
    pub fn interact(&mut self, net: &GNet, book: &Book) -> bool {
        let redex = self.rbag.pop_redex(net);
        if redex.is_zero() {
            return true;
        }

        // Gets redex ports A and B.
        let mut a = redex.get_fst();
        let mut b = redex.get_snd();

        // Gets the rule type.
        let mut rule = Port::get_rule(a, b);

        // Used for root redex.
        if a.get_tag() == REF && b == ROOT {
            rule = CALL;
        // Swaps ports if necessary.
        } else if Port::should_swap(a, b) {
            mem::swap(&mut a, &mut b);
        }

        let success = match rule {
            LINK => self.interact_link(net, a, b),
            CALL => self.interact_call(net, a, b, book),
            VOID => self.interact_void(net, a, b),
            ERAS => self.interact_eras(net, a, b),
            ANNI => self.interact_anni(net, a, b),
            COMM => self.interact_comm(net, a, b),
            OPER => self.interact_oper(net, a, b),
            SWIT => self.interact_swit(net, a, b),
            _ => panic!("Invalid rule"),
        };

        // pushes redex back if there is an error, (assuming retrying later is safe?)
        // otherwise increment the interaction count.
        if !success {
            self.rbag.push_redex(net, redex);
            panic!(
                "FAIL: {rule}\n{}\n{}\n{}={}\n{}={}\n\n",
                redex.show(),
                self.rbag.show(),
                a.get_val(),
                net.node_load(a.get_val() as usize).show(),
                b.get_val(),
                net.node_load(b.get_val() as usize).show(),
            );
        } else if rule != LINK {
            self.itrs += 1;
        }

        return true;
    }

    pub fn evaluator(&mut self, net: &GNet, book: &Book) {
        let mut tick = 0;
        let mut sidx = 0;
        let mut busy = self.tid == 0;

        loop {
            // Increments the tick
            tick += 1;
            // If we have redexes...
            if self.rbag.len() > 0 {
                // update global idle counter
                if !busy {
                    net.idle.fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
                    busy = true;
                }
                // Pops a redex and perform an interaction
                self.interact(net, book);
            } else {
                // update global idle counter
                if busy {
                    net.idle.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                    busy = false;
                }

                // find a redex to steal in another thread space
                let sid = (self.tid + 1) % TPC;
                let idx = (sid * RLEN) + sidx;
                sidx += 1;

                // try to steal the redex
                let redex = net.rbag_exchange(idx, Pair(0));
                if !redex.is_zero() {
                    //eprintln!(
                    //    "[{:04x}] stole task {} from {:04x} | itrs={} idle={} | {}",
                    //    self.tid,
                    //    idx,
                    //    sid,
                    //    self.itrs,
                    //    net.idle.load(core::sync::atomic::Ordering::Relaxed),
                    //    redex.show(),
                    //);
                    self.rbag.push_redex(net, redex);
                    continue;
                } else {
                    //eprintln!(
                    //    "[{:04x}] failed to steal task {} from {:04x} | itrs={} idle={}",
                    //    self.tid,
                    //    idx,
                    //    sid,
                    //    self.itrs,
                    //    net.idle.load(core::sync::atomic::Ordering::Relaxed),
                    //);
                    sidx = 0;
                }

                // lets just wait on some other threads.
                std::thread::yield_now();
                // check if all threads are idle after a good amount of time
                // TODO: how lower this without having threads exit early
                if tick % 256000 == 0
                    && net.idle.load(core::sync::atomic::Ordering::Relaxed) == TPC as _
                {
                    break;
                }
            }
        }

        net.itrs.fetch_add(self.itrs as u64, Ordering::Relaxed);
        // eprintln!("[{:04x}] finished | itrs={}", self.tid, self.itrs);
        self.itrs = 0;
    }
}

// Serialization
// -------------

impl Book {
    pub fn to_buffer(&self, buf: &mut Vec<u8>) {
        // Writes the number of defs
        buf.extend_from_slice(&(self.defs.len() as u32).to_ne_bytes());

        // For each def
        for (fid, def) in self.defs.iter().enumerate() {
            // Writes the safe flag
            buf.extend_from_slice(&(fid as u32).to_ne_bytes());

            // Writes the name
            // TODO: store as varlen to save space
            let name_bytes = def.name.as_bytes();
            if name_bytes.len() < 256 {
                buf.extend_from_slice(&name_bytes[..name_bytes.len()]);
                buf.resize(buf.len() + (256 - name_bytes.len()), 0);
            } else {
                panic!("Name too long: {}", def.name);
            }

            // Writes the safe flag
            buf.extend_from_slice(&(def.safe as u32).to_ne_bytes());

            // Writes the rbag length
            buf.extend_from_slice(&(def.rbag.len() as u32).to_ne_bytes());

            // Writes the node length
            buf.extend_from_slice(&(def.node.len() as u32).to_ne_bytes());

            // Writes the vars length
            buf.extend_from_slice(&(def.vars as u32).to_ne_bytes());

            // Writes the root
            buf.extend_from_slice(&def.root.0.to_ne_bytes());

            // Writes the rbag buffer
            for pair in &def.rbag {
                buf.extend_from_slice(&pair.0.to_ne_bytes());
            }

            // Writes the node buffer
            for pair in &def.node {
                buf.extend_from_slice(&pair.0.to_ne_bytes());
            }
        }
    }
}

// Readback
// --------

pub struct Ctr {
    pub tag: u32,
    pub args_len: usize,
    pub args_buf: [Port; 16],
}
// Reads back a λ-Encoded constructor from device to host.
// Encoding: λt ((((t TAG) arg0) arg1) ...)
impl Ctr {
    pub fn from(net: &GNet, book: &Book, port: Port) -> Self {
        let mut ctr = Self {
            args_len: 0,
            args_buf: [FREE; 16],
            tag: u32::MAX,
        };
        // Loads root lambda
        let lam_port = net.expand(book, port);
        if lam_port.get_tag() != CON {
            return ctr;
        }
        let lam_node = net.node_load(lam_port.get_val() as usize);
        // Loads first application
        let mut app_port = net.expand(book, lam_node.get_fst());
        if app_port.get_tag() != CON {
            return ctr;
        };
        let mut app_node = net.node_load(app_port.get_val() as usize);

        // Loads first argument (as the tag)
        let mut arg_port = net.expand(book, app_node.get_fst());
        if arg_port.get_tag() != NUM {
            return ctr;
        };

        ctr.tag = Numb(arg_port.get_val()).get_u24();

        // Loads remaining arguments
        loop {
            app_port = net.expand(book, app_node.get_snd());
            if app_port.get_tag() != CON {
                break;
            }
            app_node = net.node_load(app_port.get_val() as usize);
            arg_port = net.expand(book, app_node.get_fst());
            ctr.args_buf[ctr.args_len] = arg_port;
            ctr.args_len += 1;
        }

        ctr
    }
}

// Debug
// -----

impl Port {
    pub fn show(&self) -> String {
        match self.get_tag() {
            VAR => format!("VAR:{:08X}", self.get_val()),
            REF => format!("REF:{:08X}", self.get_val()),
            ERA => format!("ERA:{:08X}", self.get_val()),
            NUM => format!("NUM:{:08X}", self.get_val()),
            CON => format!("CON:{:08X}", self.get_val()),
            DUP => format!("DUP:{:08X}", self.get_val()),
            OPR => format!("OPR:{:08X}", self.get_val()),
            SWI => format!("SWI:{:08X}", self.get_val()),
            _ => panic!("Invalid tag"),
        }
    }
}

impl Pair {
    pub fn show(&self) -> String {
        format!("{} ~ {}", self.get_fst().show(), self.get_snd().show())
    }
}

impl RBag {
    pub fn show(&self) -> String {
        let mut s = String::new();
        s.push_str("RBAG | FST-TREE     | SND-TREE    \n");
        s.push_str("---- | ------------ | ------------\n");
        for (i, pair) in self.hi.iter().enumerate() {
            s.push_str(&format!(
                "{:04X} | {} | {}\n",
                i,
                pair.get_fst().show(),
                pair.get_snd().show()
            ));
        }
        s.push_str("~~~~ | ~~~~~~~~~~~~ | ~~~~~~~~~~~~\n");
        for (i, pair) in self.lo.iter().enumerate() {
            s.push_str(&format!(
                "{:04X} | {} | {}\n",
                i + self.hi.len(),
                pair.get_fst().show(),
                pair.get_snd().show()
            ));
        }
        s.push_str("==== | ============ | ============\n");
        return s;
    }
}

impl<'a> GNet<'a> {
    pub fn show(&self) -> String {
        let mut s = String::new();
        s.push_str("NODE | FST-PORT     | SND-PORT     \n");
        s.push_str("---- | ------------ | ------------\n");
        //for i in 0..256 {
        for i in 0..self.nlen - 1 {
            let node = self.node_load(i);
            if node.0 != 0 {
                s.push_str(&format!(
                    "{:04X} | {} | {}\n",
                    i,
                    node.get_fst().show(),
                    node.get_snd().show()
                ));
            }
        }
        s.push_str("==== | ============ | ============\n");
        s.push_str("VARS | VALUE        |\n");
        s.push_str("---- | ------------ |\n");
        //for i in 0..256 {
        for i in 0..self.vlen - 1 {
            let var = self.vars_load(i);
            if var.0 != 0 {
                s.push_str(&format!("{:04X} | {} |\n", i, var.show()));
            }
        }
        let root = self.vars_load(0x1FFFFFFF);
        s.push_str(&format!("ROOT | {} |\n", root.show()));
        s.push_str("==== | ============ |\n");
        return s;
    }
}

impl Book {
    pub fn show(&self) -> String {
        let mut s = String::new();
        for def in &self.defs {
            s.push_str(&format!(
                "==== | ============ | ============ {} (vars={},safe={})\n",
                def.name, def.vars, def.safe
            ));
            s.push_str("NODE | FST-PORT     | SND-PORT     \n");
            s.push_str("---- | ------------ | ------------\n");
            for (i, node) in def.node.iter().enumerate() {
                s.push_str(&format!(
                    "{:04X} | {} | {}\n",
                    i,
                    node.get_fst().show(),
                    node.get_snd().show()
                ));
            }
            s.push_str("==== | ============ | ============\n");
            s.push_str("RBAG | FST-TREE     | SND-TREE    \n");
            s.push_str("---- | ------------ | ------------\n");
            for (i, node) in def.rbag.iter().enumerate() {
                s.push_str(&format!(
                    "{:04X} | {} | {}\n",
                    i,
                    node.get_fst().show(),
                    node.get_snd().show()
                ));
            }
            s.push_str("==== | ============ | ============\n");
        }
        return s;
    }
}

impl Book {
    // Creates a demo program that is equivalent to:
    //   lop  = λn switch n { 0: 0; _: (lop n-1) }
    //   fun  = λn switch n { 0: (lop LOOPS); _: (+ (fun n-1) (fun n-1)) }
    //   main = (fun DEPTH)
    // Or, in core syntax:
    //   @fun  = (?<(@fun0 @fun1) a> a)
    //   @fun0 = a & @lop ~ (#65536 a)
    //   @fun1 = ({a b} c) & @fun ~ (a <+ d c>) & @fun ~ (b d)
    //   @lop  = (?<(#0 @lop0) a> a)
    //   @lop0 = (a b) & @lop ~ (a b)
    //   @main = a & @fun ~ (#10 a)
    //pub fn new_demo(depth: u32, loops: u32) -> Self {
    //let fun = Def {
    //name: "fun".to_string(),
    //safe: true,
    //rbag: vec![],
    //node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x1F),Port(0x00)), Pair::new(Port(0x09),Port(0x11)), Pair::new(Port(0x14),Port(0x00))],
    //vars: 1,
    //};
    //let fun0 = Def {
    //name: "fun0".to_string(),
    //safe: true,
    //rbag: vec![Pair::new(Port(0x19),Port(0x0C))],
    //node: vec![Pair::new(Port(0x00),Port(0x00)), Pair::new(Port::new(NUM,loops),Port(0x00))],
    //vars: 1,
    //};
    //let fun1 = Def {
    //name: "fun1".to_string(),
    //safe: false,
    //rbag: vec![Pair::new(Port(0x01),Port(0x1C)), Pair::new(Port(0x01),Port(0x2C))],
    //node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x15),Port(0x10)), Pair::new(Port(0x00),Port(0x08)), Pair::new(Port(0x00),Port(0x26)), Pair::new(Port(0x18),Port(0x10)), Pair::new(Port(0x08),Port(0x18))],
    //vars: 4,
    //};
    //let lop = Def {
    //name: "lop".to_string(),
    //safe: true,
    //rbag: vec![],
    //node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x1F),Port(0x00)), Pair::new(Port(0x03),Port(0x21)), Pair::new(Port(0x14),Port(0x00))],
    //vars: 1,
    //};
    //let lop0 = Def {
    //name: "lop0".to_string(),
    //safe: true,
    //rbag: vec![Pair::new(Port(0x19),Port(0x14))],
    //node: vec![Pair::new(Port(0x0C),Port(0x00)), Pair::new(Port(0x00),Port(0x08)), Pair::new(Port(0x00),Port(0x08))],
    //vars: 2,
    //};
    //let main = Def {
    //name: "main".to_string(),
    //safe: true,
    //rbag: vec![Pair::new(Port(0x01),Port(0x0C))],
    //node: vec![Pair::new(Port(0x00),Port(0x00)), Pair::new(Port::new(NUM,depth),Port(0x00))],
    //vars: 1,
    //};
    //return Book {
    //defs: vec![fun, fun0, fun1, lop, lop0, main],
    //};
    //}
}

#[test]
fn test_f24() {
    // Test that numbers in range round-trip correctly:
    let min_positive = f32::from_bits(0b0_00000000_000000000000001_00000000);
    let max_subnormal = f32::from_bits(0b0_00000000_111111111111111_00000000);
    let min_normal = f32::from_bits(0b0_00000001_000000000000000_00000000);
    for x in [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.1000061,
        -1.1000061,
        f32::NAN,
        f32::NEG_INFINITY,
        f32::INFINITY,
        min_positive,
        -min_positive,
        min_positive * 123.0,
        -min_positive * 123.0,
        max_subnormal,
        min_normal,
    ] {
        let y = Numb::new_f24(x).get_f24();
        assert!(x.is_nan() && y.is_nan() || x == y);
    }

    for (i, o) in [
        // Test rounding ties to even:
        (f32::from_bits(0b00_00000000), f32::from_bits(0b00_00000000)),
        (f32::from_bits(0b00_01111111), f32::from_bits(0b00_00000000)),
        (f32::from_bits(0b00_10000000), f32::from_bits(0b00_00000000)),
        (f32::from_bits(0b00_10000001), f32::from_bits(0b01_00000000)),
        (f32::from_bits(0b00_11111111), f32::from_bits(0b01_00000000)),
        (f32::from_bits(0b01_00000000), f32::from_bits(0b01_00000000)),
        (f32::from_bits(0b01_01111111), f32::from_bits(0b01_00000000)),
        (f32::from_bits(0b01_10000000), f32::from_bits(0b10_00000000)),
        (f32::from_bits(0b01_10000001), f32::from_bits(0b10_00000000)),
        (f32::from_bits(0b01_11111111), f32::from_bits(0b10_00000000)),
    ] {
        assert_eq!(Numb::new_f24(i).get_f24(), o);
    }

    // Test that NaNs are not turned into infinities
    assert!(
        Numb::new_f24(f32::from_bits(0b0_11111111_000000000000000_00000001))
            .get_f24()
            .is_nan()
    );
    assert!(
        Numb::new_f24(f32::from_bits(0b1_11111111_000000000000000_00000001))
            .get_f24()
            .is_nan()
    );
    assert!(
        Numb::new_f24(f32::from_bits(0b0_11111111_111111111111111_11111111))
            .get_f24()
            .is_nan()
    );
}
