{
  description = "A massively parallel functional runtime.";
  inputs = {
    flake-compat = {
      flake = false;
      url = "github:edolstra/flake-compat";
    };
    nci = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:yusdacra/nix-cargo-integration";
    };
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs = inputs: let
    nix-filter = import inputs.nix-filter;
    pkgs = common: packages: builtins.map (element: common.pkgs.${element}) packages;
  in
    inputs.nci.lib.makeOutputs {
      config = common: {
        cCompiler = {
          enable = true;
          package = common.pkgs.clang;
        };
        outputs = {
          defaults = {
            app = "hvm";
            package = "hvm";
          };
        };
        runtimeLibs = pkgs common ["openssl"];
        shell = {commands = builtins.map (element: {package = common.pkgs.${element};}) ["ghc" "nodejs"];};
      };
      pkgConfig = common: let
        override = {buildInputs = pkgs common ["openssl" "pkg-config"];};
      in {
        hvm = {
          app = true;
          build = true;
          overrides = {inherit override;};
          depsOverrides = {inherit override;};
          profiles = {
            dev = false;
            dev_fast = false;
            release = false;
          };
        };
      };
      # Only include directories necessary for building the project, to make the derivation smaller.
      root = nix-filter {
        root = ./.;
        include = [
          ./src
          ./Cargo.lock
          ./Cargo.toml
          ./rust-toolchain.toml
        ];
      };
    };
}