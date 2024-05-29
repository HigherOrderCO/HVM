{
  description = "hvm2";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.fenix.url = "github:nix-community/fenix/monthly";

  outputs = { self, nixpkgs, flake-utils, fenix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        hoc-upload = pkgs.writeShellScriptBin "hoc-upload" ''
          scp src/hvm.cu hoc@192.168.194.38:~/enricozb/hvm/src/hvm.cu
        '';
      in {
        devShells.default = pkgs.mkShell {
          packages = with fenix.packages.${system}; [
            minimal.rustc
            minimal.cargo
            pkgs.clippy

            hoc-upload
          ];
        };
      });
}
