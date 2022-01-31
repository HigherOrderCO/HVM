const fs = require("fs");
const { exec, execSync } = require("child_process");
const path = require("path");

var dir = __dirname;

const RUNS = 3; // how many times run the same test

function range(init, step, size) {
  var arr = [];
  for (var i = 0; i < size; ++i) {
    arr.push(init);
    init = init + step;
  }
  return arr;
}

const programs = {
  //TreeSum: { vals: range(0, 1, 33) },
  // Composition: { vals: range(0, 1, 33) },
  // QuickSort: { vals: range(0, 1, 129) },
  //LambdaArithmetic: { vals: range(0, 1, 123) },
  // ListFold: { vals: range(0, 1, 65) },
  Fibonacci: { vals: range(0, 1, 43) }
};

const evaluators = {
  //HvmInterpreter: {
    //pre: (name, file_path, temp_dir) => [],
    //execution: (name, n, temp_dir) => `hvm run ${name} ${n}`,
    //extension: ".hvm",
  //},
  RUST: {
    pre: (name, file_path) => [`rustc -O ${name} -o ${dir}/.bin/rust`],
    execution: (name, n) => `${dir}/.bin/rust ${n}`,
    extension: ".rs"
  },
  // JS: {
  //   pre: (name, file_path) => [],
  //   execution: (name, n) => `node ${name} ${n}`,
  //   extension: ".js"
  // },
  HVM: {
    pre: (name, file_path) => ["hvm compile " + name, `clang -O2 ${file_path}/main.c -o ${dir}/.bin/hvm`],
    execution: (name, n) => `${dir}/.bin/hvm ${n}`,
    extension: ".hvm",
  },
  GHC: {
    pre: (name, file_path) => [`ghc -O2 ${name} -o ${dir}/.bin/ghc`],
    execution: (name, n) => `${dir}/.bin/ghc ${n}`,
    extension: ".hs",
  },
}

function run_pre_commands(evaluator, file_name, file_path) {
  try {
    // get pre-commands for the environment
    const pres = evaluator.pre(file_name, file_path);
    // runs all pre-commands, if any
    for (pre_command of pres) {
      execSync(pre_command);
    }
    
  } catch(e) {
    console.log(e);
    throw "Error while running pre commands";
  }
}

function run_execution(evaluator, file_name, ctx) {
  let tests_perf = [];
  var total_time = 0;
  for (let i = 0; i < RUNS+1; i++) {
    const command = evaluator.execution(file_name, ctx.n);
    //console.log(command);
    // exec evaluator and measure its time
    let start = performance.now();
    execSync(command);
    let end = performance.now();

    // calculate, show and store time
    let time = end - start;
    console.log(`Time ${i}: ${time}`);

    if (i > 0) { // ignore first run
      total_time += time;
      //test_results.push({
        //test: ctx.program_name,
        //targ: ctx.evaluator_name,
        //argm: ctx.n,
        //time: time,
      //});
    }
  }

  ctx.result.push({
    targ: ctx.evaluator_name,
    prog: ctx.program_name,
    time: total_time / RUNS / 1000,
    argm: ctx.n,
  });

  return tests_perf;
}

function run_n (ctx) {
  const {file_path, evaluator_name, n} = ctx;
  const abs_file_path = path.join(dir, file_path);
  const evaluator = evaluators[evaluator_name];

  const file_name = path.join(abs_file_path, "main" + evaluator.extension);
  run_pre_commands(evaluator, file_name, abs_file_path);

  // consoles
  console.log("===========================")
  console.log(`[${evaluator_name}] ${file_path} ${n}`);

  // process.chdir(temp_dir);
  run_execution(evaluator, file_name, ctx);
  // process.chdir("..");
}

function main() {
  // for each program
  for (let program_name in programs) {
    var file_path = program_name;
    var program = programs[program_name];

    // will store the results
    let result = [];

    // for each evaluator enviroment
    for (evaluator_name in evaluators) {
      try {
        // for each n value
        for (var n of program.vals) {
          //console.log("-", n, file_path, evaluator_name, n, RUNS, result, temp_dir);
          run_n({program_name, file_path, evaluator_name, n, RUNS, result});
        }
      } catch(e) {
        console.log("Could not run for " + file_path + ": " + evaluator_name + " target. Verify if it exist.");
        console.log("Details: ", e);
      }
    }

    // Saves results.
    // TODO: messy code, improve
    console.log("Done! Saving results...");
    var charts = {};
    for (var name in programs) {
      charts[name] = {X: programs[name].vals.slice(0)};
      for (var eva in evaluators) {
        charts[name][eva] = [];
      }
    }
    for (var res of result) {
      charts[res.prog][res.targ].push(res.time);
    }
    var csvs = [];
    var evas = Object.keys(evaluators);
    for (var prog in charts) {
      var chart = charts[prog];
      var rows = [["X"].concat(evas)];
      for (var i = 0; i < chart.X.length; ++i) {
        var row = [chart.X[i]];
        for (var eva of evas) {
          row.push(chart[eva][i]);
        }
        rows.push(row);
      }
      var text = rows.map(row => row.join(",")).join("\n");
      csvs.push({prog, text}); 
    }
    for (var {prog, text} of csvs) {
      fs.writeFileSync(path.join(dir, "_results_", prog+".csv"), text);
    }
    console.log("Results saved.");
  };

  // delete temp folder
  //fs.rmSync(temp_dir, {recursive: true});

}


try {
  main();
} catch(e) {
  console.log(e);
}
// console.log(exec("hvm", ()));
