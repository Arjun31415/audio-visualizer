{
  description = "cavapaper-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    nixgl = {
      url = "github:guibou/nixGL";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };

  outputs = {
    nixgl,
    nixpkgs,
    ...
  } @ inputs: let
    pkgs = import nixpkgs {
      system = "x86_64-linux";
      overlays = [nixgl.overlay];
    };
  in
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];

      perSystem = {
        config,
        pkgs,
        system,
        lib,
        ...
      }: {
        devShells.default = pkgs.mkShell {
          inputsFrom = [config.packages.cavapaper];
          packages = with pkgs; [
            cargo
            clippy
            pre-commit
            rust-analyzer
            rustc
            rustfmt
            rustPackages.clippy
            glxinfo
            taplo
          ];
          RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
        };
        packages =
          {
            cavapaper = pkgs.rustPlatform.buildRustPackage {
              pname = "cavapaper-rs";
              version = "0.1.0";
              src = ./.;
              cargoLock = {
                lockFile = ./Cargo.lock;
              };
              nativeBuildInputs = with pkgs; [
                pkg-config
                rust-bindgen
              ];
              buildInputs = with pkgs; [
                wayland
                alsa-lib
                mesa.drivers
              ];
              LD_LIBRARY_PATH = "/run/opengl-driver/lib:/run/opengl-driver/32/lib";
            };
          }
          // {default = config.packages.cavapaper;};

        formatter = pkgs.alejandra;
      };
    };
}
