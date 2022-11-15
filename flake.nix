{
  description = "A massively parallel functional runtime.";
  inputs = {
    flake-compat = {
      flake = false;
      url = "github:edolstra/flake-compat";
    };
    nci = {
      inputs.nixpkgs.follows = "nixpkgs";
      # The next commit, 5d3d4b15b7a5f2f393fe60fcdc32deeaab88d704, is broken.
      url = "github:yusdacra/nix-cargo-integration/774b49912e6ae219e20bbb39258f8a283f6a251c";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs = inputs:
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
        runtimeLibs = [common.pkgs.openssl];
        shell = {commands = builtins.map (element: {package = common.pkgs.${element};}) ["ghc" "nodejs"];};
      };
      pkgConfig = common: let
        override = {buildInputs = [common.pkgs.openssl common.pkgs.pkg-config];};
      in {
        hvm = {
          app = true;
          build = true;
          overrides = {inherit override;};
          depsOverrides = {inherit override;};
          profiles = {
            dev = false;
            dev_fast = false;
            release = true;
          };
        };
      };
      # Exclude "wasm" directory because it causes "error: attribute 'crane' missing" and "error: expected a derivation".
      root = builtins.filterSource (path: type: !(type == "directory" && baseNameOf path == "wasm")) ./.;
    };
}
