# Resources:
# https://github.com/serokell/templates/blob/master/rust-crate2nix/flake.nix
# https://nixos.wiki/wiki/Flakes#Using_flakes_project_from_a_legacy_Nix
# https://github.com/oxalica/rust-overlay#use-in-devshell-for-nix-develop
# https://nest.pijul.com/pijul/pijul:main/SXEYMYF7P4RZM.BKPQ6
# https://github.com/srid/rust-nix-template/blob/master/flake.nix
{
  description = "Beta-optimal, massively-parallel, non-garbage-collected and strongly confluent functional runtime.";
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
                value = before.rust-bin.${rustChannel}.latest.default;
              }) [
              "cargo"
              "rustc"
            ])))
        ];
        pkgs = nixpkgs.legacyPackages.${system} // {
          inherit system overlays;
        };
        buildInputs = [
          pkgs.hyperfine
          pkgs.time
          # TODO: Remove this when `generatedCargoNix` works.
          pkgs.crate2nix
        ];
        nativeBuildInputs = [
          pkgs.clang
          pkgs.ghc
        ];
        # TODO: The following is currently blocked by https://github.com/kolloch/crate2nix/issues/213 but should be a cleaner alternative to generating Cargo.nix manually once the fix is upstreamed.
        # inherit (import "${crate2nix}/tools.nix" { inherit pkgs; })
        #   generatedCargoNix;
        # cargoNix = import
        #   (generatedCargoNix {
        #     inherit name;
        #     src = ./.;
        #   })
        #   {
        #     inherit pkgs;
        #     defaultCrateOverrides = pkgs.defaultCrateOverrides // {
        #       ${name} = attrs: {
        #         inherit buildInputs nativeBuildInputs;
        #       };
        #     };
        #   };
        cargoNix = import ./Cargo.nix {
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
