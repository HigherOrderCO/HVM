# Resources:
# https://github.com/serokell/templates/blob/master/rust-crate2nix/flake.nix
# https://nixos.wiki/wiki/Flakes#Using_flakes_project_from_a_legacy_Nix
# https://github.com/oxalica/rust-overlay#use-in-devshell-for-nix-develop
# https://nest.pijul.com/pijul/pijul:main/SXEYMYF7P4RZM.BKPQ6
# https://github.com/srid/rust-nix-template/blob/master/flake.nix
{
  description = "A lazy, beta-optimal, massively-parallel, non-garbage-collected and strongly-confluent functional compilation target.";
  inputs = {
    crate2nix = {
      url = "github:kolloch/crate2nix";
      flake = false;
    };
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };
  outputs =
    { crate2nix
    , flake-compat
    , flake-utils
    , nixpkgs
    , rust-overlay
    , self
    , ...
    }:
    let
      name = "hvm";
      rustChannel = "stable";
      rustVersion = "latest";
      inherit (builtins)
        attrValues
        listToAttrs
        map
        ;
    in
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [
          (import rust-overlay)
          (after: before:
            (listToAttrs (map
              (element: {
                name = element;
                value = before.rust-bin.${rustChannel}.${rustVersion}.default;
              }) [
              "cargo"
              "rustc"
            ])))
        ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        buildInputs = [
          pkgs.hyperfine
          pkgs.nodejs
          pkgs.time
        ];
        nativeBuildInputs = [
          pkgs.clang
          pkgs.ghc
        ];
        inherit (import "${crate2nix}/tools.nix" { inherit pkgs; })
          generatedCargoNix;
        cargoNix = import
          (generatedCargoNix {
            inherit name;
            src = ./.;
          })
          {
            inherit pkgs;
            defaultCrateOverrides = pkgs.defaultCrateOverrides // {
              ${name} = attrs: {
                inherit buildInputs nativeBuildInputs;
              };
            };
          };
      in
      {
        packages.${name} = cargoNix.rootCrate.build;
        # `nix build`
        defaultPackage = self.packages.${system}.${name};
        # `nix flake check`
        checks.${name} = cargoNix.rootCrate.build.override {
          runTests = true;
        };
        # `nix run`
        apps.${name} = flake-utils.lib.mkApp {
          inherit name;
          drv = self.packages.${system}.${name};
        };
        defaultApp = self.apps.${system}.${name};
        # `nix develop`
        devShell = pkgs.mkShell {
          inputsFrom = attrValues self.packages.${system};
          inherit buildInputs nativeBuildInputs;
        };
      }
    );
}
