// use std::fmt;

enum Term {
    Var{name: String},
    Dup{nam0: String, nam1: String, expr: BTerm, body: BTerm},
    Let{name: String, expr: BTerm, body: BTerm},
    Lam{name: String, body: BTerm},
    App{func: BTerm, argm: BTerm},
    Ctr{name: String, args: Vec<Term>},
    U32{numb: u32},
    Op2{oper: Oper, val0: BTerm, val1: BTerm},
}

type BTerm = Box<Term>;

enum Oper{
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    AND,
    OR,
    XOR,
    SHL,
    SHR,
    LTN,
    LTE,
    EQL,
    GTE,
    GTN,
    NEQ,
}

struct Rule {
    lhs: Term,
    rhs: Term,
}

// impl fmt::Display for Oper {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         use Oper::*;
//         let txt = match *self {

//         }
//         write!(f, "");
//     }
// }
