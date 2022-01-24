Usage (Nix)
-----------

#### 1. Build

[Install Nix](https://nixos.org/manual/nix/stable/installation/installation.html) then, in a shell, run:

```sh
git clone git@github.com:Kindelia/HVM
cd HVM
nix build
```

#### 2. Create an HVM file

[Same as step 2 of the "Usage" section](./README.md#2-create-a-hvm-file).

#### 3. Run it

* Interpreted:

    ```sh
    nix run . -- run main.hvm
    ```

* Compiled:

    ```sh
    nix run . -- c main.hvm            # compiles hvm to C
    nix develop                        # initialises nix development shell
    # Then in the dev shell, run:
    clang -O2 main.c -o main -lpthread # compiles C to executable
    ./main                             # runs executable
    ```
