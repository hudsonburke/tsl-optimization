{
  description = "Tendon slack length optimization";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };

          tools = with pkgs; [
            pyright
            ruff
            uv
            git
          ];
        in
        {
          default = pkgs.mkShell {
            packages = tools;
            shellHook = ''
              echo "tsl-optimization dev shell"
              echo "  uv sync           — install deps"
              echo "  uv run pytest -q   — run tests"
            '';
          };
        });
    };
}
