{
  description = "demo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nci = {
      url = "github:yusdacra/nix-cargo-integration";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    wgsl_analyzer = {
      url = "github:wgsl-analyzer/wgsl-analyzer";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ {flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      imports = [
        inputs.nci.flakeModule
      ];
      systems = ["x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin"];
      perSystem = {
        config,
        self',
        inputs',
        pkgs,
        system,
        ...
      }: let
        outputs = config.nci.outputs;
        projects = {
          new_voxel_testing = {
            path = ./.;
            drvConfig.mkDerivation.nativeBuildInputs = with pkgs; [mold pkg-config];
            drvConfig.mkDerivation.buildInputs = with pkgs; [alsa-lib libudev-zero wayland];
          };
        };
      in {
        nci = {
          inherit projects;
        };
        packages.outputs = outputs;
        devShells.default = outputs.new_voxel_testing.devShell.overrideAttrs (old: {
          packages = with pkgs; (old.packages or []) ++ [cargo-expand gdb cargo-udeps curl jq zstd just inputs.wgsl_analyzer.packages.${system}.default cargo-criterion];
        });
        formatter = pkgs.alejandra;
      };
      flake = {
        # The usual flake attributes can be defined here, including system-
        # agnostic ones like nixosModule and system-enumerating ones, although
        # those are more easily expressed in perSystem.
      };
    };
}
