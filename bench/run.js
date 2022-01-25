const fs = require("fs");
const { exec, execSync } = require("child_process");
const path = require("path");

var dir = __dirname;

const runners = {
  C: {
    first_line: (n) => `const int N = ${n};`,
    pre: (name) => ["clang -O2 "+ name +" -o bench"],
    execution: (name) => "./bench",
    extension: ".c",
  },
  HovmInterpreter: {
    first_line: (n) => `(N) = ${n}`,
    pre: (name) => [],
    execution: (name) => "hovm run " + name,
    extension: ".hovm",
  },
  HovmCompile: {
    first_line: (n) => `(N) = ${n}`,
    pre: (name) => ["hovm compile " + name, "gcc " + name + ".out.c -o bench"],
    execution: (name) => "./bench",
    extension: ".hovm",
  },
  Haskell: {
    first_line: (n) => `n          = ${n} :: Int`,
    pre: (name) => ["ghc -O2 "+ name +" -o bench"],
    execution: (name) => "./bench",
    extension: ".hs",
  },
}

function format_test_name(str, default_value) {
  if (!str) {
    return default_value;
  }
  if (str[0] === "[") {
    let values = str.substr(1, str.length - 2).split(",").map(x => x.trim());
    return values;
  }
  throw "Couldn't parse tests names";
}

function format_n_values(str, default_value) {
  if (!str) {
    return default_value;
  } else if (str[0] === "[") {
    return JSON.parse(str);
  }

  throw "Couldn't parse n_values";
}

function replace_first_line(data, runner, n) {
  return data.toString().replace(/.*\n/, runner.first_line(n) + "\n");
}

function get_params() {
  const paths    = format_test_name(process.argv[2], ["Fibonacci/fibonacci"]);
  const n_values = format_n_values(process.argv[3], [[10, 20, 30]]);
  const times    = process.argv[4] || 5;

  return {
    paths,
    n_values,
    times
  }
}

function get_file_content(runner, file_path, n) {
  try {
    const file_path_ext = file_path + runner.extension;
    const file_content = fs.readFileSync(path.join(dir, file_path_ext));
    return replace_first_line(file_content, runner, n);
  } catch(e) {
    throw "Error while reading the file, verify if it exists";
  }
}

function generate_bench_file(runner, temp_dir, file_content) {
  try {
    process.chdir(temp_dir); // change dir to a temporary folder
    const gen_file_name = "tmp" + runner.extension; // generate a test file with "tmp.xy" name in .tmp folder
    fs.writeFileSync(gen_file_name, file_content);
    return gen_file_name;
  } catch(e) {
    throw "Error while generating tmp file";
  }
}

function run_pre_commands(runner, file_name) {
  try {
    // get pre-commands for the environment
    const pres = runner.pre(file_name);
    // runs all pre-commands, if any
    for (pre_command of pres) {
      execSync(pre_command);
    }
  } catch(e) {
    throw "Error while running pre commands";
  }
}

function run_execution(runner, file_name, times) {
  let tests_perf = [];
  let command = runner.execution(file_name);
  for (let i = 0; i < times; i++) {
    // exec runner and measure its time
    let start = performance.now();
    execSync(command);
    let end = performance.now();

    // calculate, show and store time
    let time = end - start;
    console.log(`Time ${i}: ${time}`);
    tests_perf.push(time);
  }

  return tests_perf;
}

function run_n (ctx, temp_dir) {
  const {path, runner_name, n, times} = ctx;
  const runner = runners[runner_name];

  const file_content = get_file_content(runner, path, n);
  const gen_file_name = generate_bench_file(runner, temp_dir, file_content);
  
  run_pre_commands(runner, gen_file_name);

  // consoles
  console.log("===========================")
  console.log(`${runner_name}: running ${path} with n = ${n}`);
  console.log();

  const tests_perf = run_execution(runner, gen_file_name, times);
  process.chdir("..");
  return tests_perf;
}

function main() {
  const params = get_params();
  const {paths, n_values, times} = params;

  // will store the results
  let result = {};
  // create a temp folder
  var temp_dir = fs.mkdtempSync(path.join(dir, ".tmp-"));

  // for each test
  paths.forEach((path, i) => {
    result[path] = {};
    // for each runner enviroment
    for (runner_name in runners) {
      result[path][runner_name] = {};
      try {
        // for each n value
        for (n of n_values[i]) {
          const tests_perf = run_n({path, runner_name, n, times}, temp_dir);
          result[path][runner_name][n] = tests_perf;
        }
      } catch(e) {
        console.log("Could not run for " + path + ": " + runner_name + " target. Verify if it exist.");
        console.log("Details: ", e);
      }
    }
  });

  // write result
  const result_json = JSON.stringify(result);
  fs.writeFileSync("result.json", result_json);
  // delete temp folder
  fs.rmSync(temp_dir, {recursive: true});
}


try {
  main();
} catch(e) {
  console.log(e);
}
// console.log(exec("hovm", ()));