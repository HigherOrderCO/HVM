Usage (Nix)
-----------

#### 1. Build HVM

[Install Nix](https://nixos.org/manual/nix/stable/installation/installation.html) and [Nix Flakes](https://nixos.wiki/wiki/Flakes#Installing_flakes) then, in a shell, run:

```sh
git clone git@github.com:Kindelia/HVM
cd HVM
# Build HVM
nix build
# Install it to your Nix profile
nix profile install
```

#### 2. Create an HVM file

[Same as step 2 of the "Usage" section](./README.md#2-create-an-hvm-file).

#### 3. Run/compile it

```sh
# Interpret the main.hvm file, passing 10 as an argument
hvm run main.hvm 10
# Compile it to C
hvm compile main.hvm
# Intialise the Nix development shell
nix develop
# Compile the resulting C code
clang -O2 main.c -o main -lpthread
# Run the resulting binary, passing 30 as an argument
./main 30
```
