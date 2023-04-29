#![allow(unused)]

use proptest::prelude::*;
use hvm::syntax::{Oper, Term, Rule};

const MAX_U60: u64 = !0 >> 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Op(Oper);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TermParams {
    pub max_identifiers: usize,
    pub max_depth: usize,
}

impl Default for TermParams {
    fn default() -> Self {
        // ensures by default there are variabes in expressions
        Self {
            max_identifiers: 10,
            max_depth: 6,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LesserTerm {
  Identifier { name: usize },
  Lam { name: usize, body: Box<LesserTerm> },
  App { func: Box<LesserTerm>, argm: Box<LesserTerm> },
  U60 { numb: u64 },
  F60 { numb: f64 },
  Op2 { oper: Op, val0: Box<LesserTerm>, val1: Box<LesserTerm> },
}

impl Arbitrary for Op {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
        use Oper::*;
        let op_arr = [Add, Sub, Mul, Div, Mod, And, Or,  Xor, Shl, Shr, Lte, Ltn, Eql, Gte, Gtn, Neq];
        (0usize..15).prop_map(move |i| Op(op_arr[i])).boxed()
    }
}

impl Arbitrary for LesserTerm {
    type Parameters = TermParams;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(params: Self::Parameters) -> Self::Strategy {
        use LesserTerm::*;
        let ident_strat = (0..(params.max_identifiers)).prop_map(|name| Identifier{ name });
        let u60_strat = (0..MAX_U60).prop_map(|numb| U60 {numb});
        // remove the last 4 bits to ensure floats match after conversion
        let f60_strat = proptest::num::f64::NORMAL.prop_map(|x| F60 {numb: f64::from_bits((x.to_bits() >> 4) << 4)});
        prop_oneof![ u60_strat, f60_strat, ident_strat].prop_recursive(
            params.max_depth as _, // No more than `max_depth` levels deep
            64, // Target around 64 total elements
            16, // Each collection is up to 16 elements long
            move |element| {
                let boxed_elem = element.prop_map(|x| Box::new(x));
                prop_oneof![
                    (0..params.max_identifiers, boxed_elem.clone()).prop_map(|(name, body)| Lam { name, body }),
                    (boxed_elem.clone(), boxed_elem.clone()).prop_map(|(func, argm)| App { func, argm }),
                    (proptest::arbitrary::any::<Op>(), boxed_elem.clone(), boxed_elem.clone()).prop_map(|(oper, val0, val1)| Op2 { oper, val0, val1 }),
                ]
        }).boxed()
    }
}

impl From<LesserTerm> for Term {
    fn from(term: LesserTerm) -> Self {
        fn from_inner(term: LesserTerm, scope: &mut Vec<usize>) -> Term {
            match term {
                LesserTerm::Identifier { name } => {
                    if scope.contains(&name) {
                        Term::variable(format!("v{name}"))
                    } else {
                        // ensures that the term can evaluate without additional rules
                        Term::integer(1)
                    }
                },
                LesserTerm::Lam { name, body } => {
                    scope.push(name);
                    let out = Term::lambda(format!("v{name}"), from_inner(*body, scope));
                    scope.pop();
                    out
                },
                LesserTerm::App { func, argm } => Term::application(from_inner(*func, scope), from_inner(*argm, scope)),
                LesserTerm::U60 { numb } => Term::integer(numb),
                LesserTerm::F60 { numb } => Term::float(numb),
                LesserTerm::Op2 { oper, val0, val1 } => Term::binary_operator(oper.0, from_inner(*val0, scope), from_inner(*val1, scope)),
            }
        }
        from_inner(term, &mut vec![])
    }
}
