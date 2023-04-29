mod arbitrary;
mod interpreter;

use std::error::Error;
use hvm::rulebook::sanitize_rule;
use proptest::prelude::*;
use hvm::syntax::{Term, Rule};

fn sanitize_term(term: &Term) -> Result<Term, Box<dyn Error + Sync + Send + 'static>> {
    let rule = Rule::new(Term::constructor("HVM_MAIN_CALL", []), term.clone());
    Ok(*sanitize_rule(&rule)?.rhs)
}

// returns true if the two terms are identical up to a renaming of variables
fn isomorphic(term1: &Term, term2: &Term) -> bool {
    // currently assumes both terms are sanitizeable
    // and that the names of sanitized variables only depend on order of apearance
    sanitize_term(term1).unwrap() == sanitize_term(term2).unwrap()
}

#[test]
fn compare_to_model() {
    let params = arbitrary::TermParams::default();
    let runtime = hvm::RuntimeBuilder::default().set_thread_count(1).build();
    proptest!(|(lesser_term in arbitrary::LesserTerm::arbitrary_with(params))| {
        let term: Term = lesser_term.into();
        let comp_result = std::panic::catch_unwind(|| {
            let runtime = &runtime; // shadow to prevent moving
            return std::thread::scope(move |s| {
                let term_copy = term.clone();
                let hvm_handle = s.spawn(move || {
                    runtime.normalize_term(&term)
                });
                let interp_handle = s.spawn(move || {
                    interpreter::complete_dups(interpreter::normalize(term_copy))
                });
                let sec = std::time::Duration::from_secs(1);
                for _ in 0..60 {
                    if !interp_handle.is_finished() {
                        std::thread::sleep(sec);
                    } else if !hvm_handle.is_finished() {
                        // HVM should have terminated by now
                        return None;
                    } else {
                        return Some((hvm_handle.join(), interp_handle.join()));
                    }
                }
                panic!("term (possibly) does not terminate");
            });
        });

        if let Ok(comp) = comp_result {
            match comp {
                Some((Ok(output), Ok(expected_output))) => assert!(isomorphic(&output, &expected_output), "expected {}, got {}", expected_output, output),
                Some((Err(err), Ok(expected_output))) => {
                    eprintln!("expected result {}", expected_output);
                    std::panic::resume_unwind(err);
                },
                None => panic!("HVM did not terminate"),
                _ => {},
            }
        }
    })
}
