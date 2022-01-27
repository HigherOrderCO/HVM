const fs = require("fs");
const { exec, execSync } = require("child_process");
const path = require("path");

var dir = __dirname;

const runners = {
  // C: {
  //   first_line: (n) => `const int N = ${n};`,
  //   pre: (name) => ["clang -O2 "+ name +" -o bench"],
  //   execution: (name) => "./bench",
  //   extension: ".c",
  // },
  HovmInterpreter: {
    pre: (name, temp_dir) => [],
    execution: (name, n, temp_dir) => `hovm run ${name} ${n}`,
    extension: ".hovm",
  },
  // HovmCompile: {
  //   pre: (name, temp_dir) => ["hovm compile " + name, `clang -O2 ${name}.out.c -o ${temp_dir}/bench`],
  //   execution: (name, n, temp_dir) => `${temp_dir}/bench ${n}`,
  //   extension: ".hovm",
  // },
  // Haskell: {
  //   pre: (name, temp_dir) => ["ghc -O2 "+ name +" -o bench"],
  //   execution: (name, n, temp_dir) => `${temp_dir}/bench ${n}`,
  //   extension: ".hs",
  // },
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
  const paths    = format_test_name(process.argv[2], ["Fibonacci"]);
  const n_values = format_n_values(process.argv[3], [[10, 20, 30]]);
  const times    = process.argv[4] || 5;

  return {
    paths,
    n_values,
    times
  }
}

// function get_file_content(runner, file_path, n) {
//   console.log(file_path);
//   try {
//     const file_name = "main" + runner.extension;
//     const file_content = fs.readFileSync(path.join(dir, file_path, file_name));
//     return replace_first_line(file_content, runner, n);
//   } catch(e) {
//     throw "Error while reading the file, verify if it exists";
//   }
// }

function generate_bench_file(runner, temp_dir, file_content) {
  try {
    // process.chdir(temp_dir); // change dir to a temporary folder
    const gen_file_name = "tmp" + runner.extension; // generate a test file with "tmp.xy" name in .tmp folder
    fs.writeFileSync(gen_file_name, file_content);
    return gen_file_name;
  } catch(e) {
    throw "Error while generating tmp file";
  }
}

function run_pre_commands(runner, file_name, temp_dir) {
  try {
    // get pre-commands for the environment
    const pres = runner.pre(file_name, temp_dir);
    // runs all pre-commands, if any
    for (pre_command of pres) {
      console.log(pre_command);
      execSync(pre_command);
    }
    
  } catch(e) {
    throw "Error while running pre commands";
  }
}

function run_execution(runner, file_name, times, ctx, temp_dir) {
  let tests_perf = [];
  for (let i = 0; i < times; i++) {
    const command = runner.execution(file_name, n, temp_dir);
    console.log(command);
    // exec runner and measure its time
    let start = performance.now();
    execSync(command);
    let end = performance.now();

    // calculate, show and store time
    let time = end - start;
    console.log(`Time ${i}: ${time}`);

    ctx.result.push({
      target: ctx.runner_name,
      n: ctx.n,
      time: time,
    });
  }

  return tests_perf;
}

function run_n (ctx, temp_dir) {
  const {file_path, runner_name, n, times} = ctx;
  const runner = runners[runner_name];

  // const file_content = get_file_content(runner, path, n);
  // const gen_file_name = generate_bench_file(runner, temp_dir, file_content);
  
  const file_name = path.join(file_path, "main" + runner.extension);
  console.log("file name", file_name);
  run_pre_commands(runner, file_name, temp_dir);

  // consoles
  console.log("===========================")
  console.log(`${runner_name}: running ${file_path} with n = ${n}`);
  console.log();

  process.chdir(temp_dir);
  run_execution(runner, file_name, times, ctx , temp_dir);
  process.chdir("..");
}

function main() {
  const params = get_params();
  const {paths, n_values, times} = params;

  // create a temp folder
  var temp_dir = fs.mkdtempSync(path.join(dir, ".tmp-"));

  // for each test
  paths.forEach((file_path, i) => {
    // will store the results
    let result = [];
    // for each runner enviroment
    for (runner_name in runners) {
      try {
        // for each n value
        for (n of n_values[i]) {
          run_n({file_path, runner_name, n, times, result}, temp_dir);
        }
      } catch(e) {
        console.log("Could not run for " + file_path + ": " + runner_name + " target. Verify if it exist.");
        console.log("Details: ", e);
      }
    }

    // write result
    const result_json   = JSON.stringify(result);
    if (file_path) {
      try {
        fs.mkdirSync(["./Results", file_path].join(path.sep), {recursive: true});
      } catch(e) {

      } finally {
        console.log(file_path);
        const result_path   = ["./Results", file_path, "result.json"].join(path.sep);
        fs.writeFileSync(result_path, result_json, {recursive: true});
      }
    }
  });

  // delete temp folder
  fs.rmSync(temp_dir, {recursive: true});

}


try {
  main();
} catch(e) {
  console.log(e);
}
// console.log(exec("hovm", ()));