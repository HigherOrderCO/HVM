#![allow(clippy::identity_op)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_attributes)]
#![allow(unused_imports)]

pub mod base;
pub mod data;
pub mod rule;

use std::collections::HashMap;
use sysinfo::{System, SystemExt, RefreshKind};

pub use base::{*};
pub use data::{*};
pub use rule::{*};

use crate::language;

pub const CELLS_PER_KB: usize = 0x80;
pub const CELLS_PER_MB: usize = 0x20000;
pub const CELLS_PER_GB: usize = 0x8000000;

// If unspecified, allocates `max(16 GB, 75% free_sys_mem)` memory
pub fn default_heap_size() -> usize {
  use sysinfo::SystemExt;
  let available_memory = System::new_with_specifics(RefreshKind::new().with_memory()).free_memory();
  let heap_size = (available_memory * 3 / 4) / 8;
  let heap_size = std::cmp::min(heap_size as usize, 16 * CELLS_PER_GB);
  return heap_size as usize;
}

// If unspecified, spawns 1 thread for each available core
pub fn default_heap_tids() -> usize {
  return std::thread::available_parallelism().unwrap().get();
}

/// a builder for Runtime to determine its configuration
pub struct RuntimeBuilder {
    rules: Vec<language::syntax::Rule>,
    strictness_maps: HashMap<String, Vec<bool>>,
    functions: HashMap<String, Function>,
    thread_count: usize,
    heap_size: usize,
    debug: bool,
}

/// the runtime which evaluates the HVM code
pub struct Runtime {
    heap: Heap,
    program: Program,
    book: language::rulebook::RuleBook,
    thread_ids: Box<[usize]>,
    debug: bool,
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self {
            rules: Default::default(),
            strictness_maps: Default::default(),
            functions: Default::default(),
            thread_count: default_heap_tids(),
            heap_size: default_heap_size(),
            debug: false,
        }
    }
}

impl RuntimeBuilder {
    /// add a HVM rule to be interpreted by the runtime
    pub fn add_rule(mut self, rule: language::syntax::Rule) -> Self {
        self.rules.push(rule);
        self
    }

    /// add multiple HVM rules at once
    pub fn add_rules(mut self, rules: impl IntoIterator<Item = language::syntax::Rule>) -> Self {
        self.rules.extend(rules.into_iter());
        self
    }

    /// add a strictness mapping to a HVM rule,
    /// where the `true` values corresponds to arguments that must be evaluated strictly.
    ///
    /// unless otherwise specifier, arguments of a constructor are evaluated lazily,
    /// so this is useful for functions which cause side-effects.
    pub fn add_strictness_map(mut self, name: String, map: impl IntoIterator<Item = bool>) -> Self {
        self.strictness_maps.insert(name, map.into_iter().collect());
        self
    }

    /// adds the rules written in the given HVM source code.
    ///
    /// returns the error message if the code failed to parse.
    pub fn add_code(mut self, code: &str) -> Result<Self, String> {
        let file = language::syntax::read_file(code)?;
        self.rules.extend(file.rules);
        for (name, smap) in file.smaps {
            self.strictness_maps.insert(name, smap);
        }
        Ok(self)
    }

    /// add a function, prepared a head of time, which will be mapped to the given symbol
    ///
    /// allows precompilation of frequently used functions,
    /// and including functionality that is not part of HVM by default, such as IO
    pub fn add_function(mut self, name: String, func: Function) -> Self {
        self.functions.insert(name, func);
        self
    }

    /// sets the number of threads that will be used to reduce terms given to the runtime.
    pub fn set_thread_count(mut self, thread_count: usize) -> Self {
        self.thread_count = thread_count;
        self
    }

    /// sets the size of the heap which stores the terms evaluated by the runtime,
    /// given in the number of terms that can be stored on the heap.
    ///
    /// use [`CELLS_PER_KB`], [`CELLS_PER_MB`] and [`CELLS_PER_GB`],
    /// to add values in terms of memory size.
    pub fn set_heap_size(mut self, heap_size: usize) -> Self {
        self.heap_size = heap_size;
        self
    }

    /// causes evaluation of terms to print debug output,
    /// showing how the given term was reduced step by step.
    pub fn set_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn build(self) -> Runtime {
        let file = language::syntax::File {
            rules: self.rules,
            smaps: self.strictness_maps.into_iter().collect(),
        };

        // Converts the file to a Rulebook
        let book = language::rulebook::gen_rulebook(&file);

        // Creates the runtime program
        let mut program = Program::new();

        // Adds the interpreted functions (from the Rulebook)
        program.add_book(&book);

        // Adds the extra functions
        for (name, fun) in self.functions {
            program.add_function(name, fun);
        }

        // Creates the runtime heap
        let heap = new_heap(self.heap_size, self.thread_count);
        let thread_ids = new_tids(self.thread_count);

        Runtime {
            heap,
            program,
            book,
            thread_ids,
            debug: self.debug,
        }
    }
}

impl Runtime {
    /// reduces the term to Normal Form,
    /// meaning that applications in the term are evaluated,
    /// untill there are no more applications in the term.
    pub fn normalize_term(&self, term: &language::syntax::Term) -> language::syntax::Term {
        let tid = 0;

        let host = alloc_term(&self.heap, &self.program, tid, &self.book, term);
        let ptr = reduce(
            &self.heap,
            &self.program,
            &self.thread_ids,
            host,
            true,
            self.debug,
        );

        let output = language::readback::as_term(&self.heap, &self.program, host);

        collect(&self.heap, &self.program.aris, tid, ptr);
        *output
    }

    /// attempts to reduce the given term to the target type if possible.
    /// on failure returns the error produced in the conversion of the reduced term.
    pub fn eval_term<T, E>(&self, term: &language::syntax::Term) -> Result<T, E>
    where language::syntax::Term: TryInto<T, Error=E> {
        self.normalize_term(term).try_into()
    }

    /// returns the number graph rewrites made by the runtime,
    /// since its initialization.
    ///
    /// this serves as a measure of the computational cost of normalizing terms.
    pub fn get_rewrite_count(&self) -> usize {
        get_cost(&self.heap) as _
    }
}
