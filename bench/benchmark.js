var fs = require("fs");
var exec_sync = require("child_process").execSync;

var SMALL = false;

var langs = {
  GHC: {
    tids: [1],
    tasks: {
      "sort/bitonic": [1, 24],
      "sort/bubble": [12, 17],
      "sort/quick": [1, 24],
      "sort/radix": [1, 24],
    },
    build: (task) => {
      exec("cp ../../examples/"+task+"/main.hs main.hs");
      exec("ghc -O2 main.hs -o main.bin");
    },
    bench: (task, size, tids) => {
      return bench("./main.bin " + size);
    },
    clean: () => {
      //exec("rm *.hs");
      //exec("rm *.hi");
      //exec("rm *.o");
      //exec("rm *.bin");
    },
  },

  HVM: {
    tids: [1, 2, 4, 8],
    tasks: {
      "sort/bitonic": [1, 24],
      "sort/bubble": [12, 17],
      "sort/quick": [1, 24],
      "sort/radix": [1, 24],
    },
    build: (task) => {
      exec("cp ../../examples/"+task+"/main.hvm main.hvm");
      exec("hvm compile main.hvm");
      exec("cd main; cargo build --release; mv target/release/main ../main.bin");
    },
    bench: (task, size, tids) => {
      return bench('./main.bin run -t '+tids+' "(Main ' + size + ')"');
    },
    clean: () => {
      exec("rm main.hvm");
      exec("rm main.bin");
    },
  },
};

// Enters the work directory
if (!fs.existsSync("work")) {
  exec("mkdir work");
}
process.chdir("work");

// Runs benchmarks and collect results
var results = [];
for (var lang in langs) {
  for (var tids of langs[lang].tids) {
    //console.log(lang);
    for (var task in langs[lang].tasks) {
      langs[lang].build(task);
      var min_size = langs[lang].tasks[task][0];
      var max_size = SMALL ? min_size + 2 : langs[lang].tasks[task][1];
      for (var size = min_size; size <= max_size; ++size) {
        if (size === min_size) {
          langs[lang].bench(task, size, tids); // dry-run to heat up
        }
        var time = langs[lang].bench(task, size, tids);
        results.push({task, lang: lang+"-"+tids, size, time});
        console.log(lang + "-" + tids + " | " + task + " | " + size + " | " + time.toFixed(3) + "s");
      }
    }
  }
}

// Writes results to JSON
fs.writeFileSync("./../results.json", JSON.stringify(results, null, 2));

// Executes a command
function exec(str) {
  try {
    return exec_sync(str).toString();
  } catch (e) {
    console.log("OUT:", e.stdout.toString());
    console.log("ERR:", e.stderr.toString());
    return Infinity;
  }
}

// Benchmarks a command
function bench(cmd) {
  var ini = Date.now();
  var res = exec(cmd, {skipThrow: 1}).toString().replace(/\n/g,"");
  if (res == Infinity) { return Infinity }
  var end = Date.now();
  return (end - ini) / 1000;
}
