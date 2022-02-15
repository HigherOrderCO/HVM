Benchmarking
------------

#### Install Dependencies
* [Node.js](https://nodejs.org/en/download/)
* [GHC](https://www.haskell.org/ghc/download.html)

#### Install HVM

See [README.md](../README.md#1-install-it) in the root directory for
installation instructions.

#### Run all benchmarks

The `run.js` script runs the benchmarks that generated the images in the
README.md.

```sh
node run.js
```

Benchmarking (Nix)
------------------

#### Install HVM

See [NIX.md](../NIX.md#1-build-and-install-hvm) in the root directory for
installation instructions.

#### Initialise Nix development shell

```sh
# Go back to the root HVM directory.
cd ..
# Initialise the dev shell.
# The rest of the instructions in this section assume that you're using the dev
# shell.
nix develop
cd bench
cd <benchmark_directory>
```

#### Benchmark Haskell code:

```sh
ghc -O2 main.hs -o main
hyperfine --show-output ./main <arguments>
```

#### Benchmark HVM code:

```sh
hvm compile main.hvm
clang -O2 main.c -o main -lpthread
hyperfine --show-output ./main <arguments>
```

#### Run all benchmarks

Same as in [the corresponding step](#run-all-benchmarks) in the "Benchmarking" section above.
