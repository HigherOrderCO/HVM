Usage (Nix)
-----------

#### 1. Access/install HVM

[Install Nix](https://nixos.org/manual/nix/stable/installation/installation.html) and enable [Flakes](https://wiki.nixos.org/wiki/Flakes#Enable_flakes) then, in a shell, run:

```sh
git clone https://github.com/HigherOrderCO/HVM.git
cd HVM
# Start a shell that has the `hvm` command without installing it.
nix shell .#hvm
# Or install it to your Nix profile.
nix profile install .#hvm
```

#### 2. Create an HVM file

[Same as step 2 of the "Usage" section](./README.md#2-create-an-hvm-file).

#### 3. Run/compile it

```sh
# Interpret the main.hvm file, passing "(Main 25)" as an argument.
hvm run -f main.hvm "(Main 25)"
# Compile it to Rust.
hvm compile main.hvm
cd main
# Initialise the Nix development shell.
nix develop .#hvm
# Compile the resulting Rust code.
cargo build --release
# Run the resulting binary.
./target/release/main run "(Main 25)"
```
