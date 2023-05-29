{
  description = "A massively parallel functional runtime.";
  inputs = {
    devenv = {
      inputs.flake-compat.follows = "flake-compat";
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:cachix/devenv";
    };
    fenix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:nix-community/fenix/monthly";
    };
    flake-compat = {
      flake = false;
      url = "github:edolstra/flake-compat";
    };
    flake-parts = {
      inputs.nixpkgs-lib.follows = "nixpkgs";
      url = "github:hercules-ci/flake-parts";
    };
    nci = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:yusdacra/nix-cargo-integration";
    };
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };
  outputs = inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      imports = builtins.map (item: inputs.${item}.flakeModule) ["devenv" "nci"];
      nci.source = (import inputs.nix-filter) {
        root = ./.;
        include = [
          "cli"
          "Cargo.lock"
          "Cargo.toml"
          "hvm"
          "rust-toolchain.toml"
        ];
      };
      perSystem = {
        config,
        self',
        inputs',
        pkgs,
        system,
        ...
      }: let
        project = "hvm";
        executable_crate = "hvm-cli";
        outputs = config.nci.outputs;
        override.overrideAttrs = old: {buildInputs = (old.buildInputs or []) ++ pkgs.lib.attrsets.attrVals ["openssl" "pkg-config"] pkgs;};
      in {
        devenv.shells.default = {
          languages = {
            javascript.enable = true;
            rust = {
              enable = true;
              version = "latest";
            };
            haskell.enable = true;
          };
          packages = [config.packages.default];
        };
        nci.projects.${project} = {
          relPath = "";
          export = true;
        };
        nci.crates.${executable_crate} = {
          depsOverrides = {inherit override;};
          overrides = {inherit override;};
          profiles.release.runTests = false;
          runtimeLibs = pkgs.lib.attrsets.attrVals ["openssl"] pkgs;
        };
        packages.default = outputs.${executable_crate}.packages.release;
      };
      systems = ["x86_64-linux" "aarch64-darwin"];
    };
}
