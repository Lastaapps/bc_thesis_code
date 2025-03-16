# TODO migrate to flake/develop.nix
let
  pkgs = import <nixpkgs> { };
in
pkgs.mkShell {
  name = "PyRigi shell";

  packages =
    (with pkgs; [
      python312
      graphviz
    ])
    ++ (with pkgs.python312Packages; [
      pip
      virtualenv
      black
    ]);

  shellHook = ''
    # https://nixos.wiki/wiki/Packaging/Quirks_and_Caveats
    # fixes libstdc++ issues and libgl.so issues
    LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=${pkgs.graphviz}/lib/:$LD_LIBRARY_PATH

    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi
    source ./.venv/bin/activate


    pip install -r ${./requirements.txt}
  '';
}
