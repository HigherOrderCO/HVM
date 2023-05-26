{
  description = "A massively parallel functional runtime.";
  inputs = {
    devenv = {
      inputs.flake-compat.follows = "flake-compat";
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:cachix/devenv";
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
        include = ["Cargo.lock" "Cargo.toml" "rust-toolchain.toml" "src"];
      };
      perSystem = {
        config,
        self',
        inputs',
        pkgs,
        system,
        ...
      }: let
        crate_name = "hvm";
        crate_outputs = config.nci.outputs.${crate_name};
        override.overrideAttrs = old: {buildInputs = (old.buildInputs or []) ++ pkgs.lib.attrsets.attrVals ["openssl" "pkg-config"] pkgs;};
      in {
        devenv.shells.default = {
          languages = {
            javascript.enable = true;
            rust.enable = true;
            haskell.enable = true;
          };
          packages = [config.packages.default];
        };
        nci.projects.${crate_name}.relPath = "";
        nci.crates.${crate_name} = {
          depsOverrides = {inherit override;};
          export = true;
          overrides = {inherit override;};
          profiles.release.runTests = false;
          runtimeLibs = pkgs.lib.attrsets.attrVals ["openssl"] pkgs;
        };
        packages.default = crate_outputs.packages.release;
      };
      systems = ["x86_64-linux" "aarch64-darwin"];
    };
}
