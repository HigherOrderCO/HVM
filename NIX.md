Usage (Nix)
-----------

#### 1. Access/install HVM

[Install Nix](https://nixos.org/manual/nix/stable/installation/installation.html) and enable [Flakes](https://nixos.wiki/wiki/Flakes#Enable_flakes) then, in a shell, run:

```sh
git clone https://github.com/Kindelia/HVM.git
cd HVM
# Start a shell that has the `hvm` command without installing it.
nix shell .
# Or install it to your Nix profile.
nix profile install .
```

#### 2. Create an HVM file

[An example program can be found here](./guide/README.md#first-program).

#### 3. Run/compile it

```sh
# Interpret the hvm file while passing an argument.
hvm run -f BMI.hvm "(BMI 62.0 1.70)"
# Compile it to Rust.
hvm compile BMI.hvm
cd BMI
# Initialise the Nix development shell.
nix develop
# Compile the resulting Rust code.
cargo build --release
# Run the resulting binary with the same argument.
./target/release/BMI "(BMI 62.0 1.70)"
```
