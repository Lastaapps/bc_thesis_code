# TODO migrate to flake/develop.nix
let
  pkgs = import <nixpkgs> { };
in
pkgs.mkShell {
  name = "PyRigi shell";

  packages =
    (with pkgs; [ python312 ])
    ++ (with pkgs.python312Packages; [
      pip
      virtualenv
      black
    ]);

  shellHook = ''
    # https://nixos.wiki/wiki/Packaging/Quirks_and_Caveats
    # fixes libstdc++ issues and libgl.so issues
    LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/:$LD_LIBRARY_PATH

    if [ ! -d .venv ]; then
      python3 -m venv .venv
    fi
    source ./.venv/bin/activate


    pip install \
    \
    pandas~=2.2.0 \
    numpy~=2.1.0 \
    scipy~=1.14.0 \
    sympy~=1.13.0 \
    matplotlib~=3.9.0 \
    \
    networkx~=3.4.0 \
    \
    pytest~=8.3.0 \
    pytest-benchmark~=5.1.0 \
    pytest-timeout~=2.3.0 \
    \
    tqdm~=4.67.0 \
    urllib3~=2.2.0 \
    distinctipy~=1.3.0 \
    \
    ipywidgets~=8.1.0 \
    ipycanvas~=0.13.0 \
    ipyevents~=2.0.0 \
    \
    notebook~=7.2.0 \
    ipython~=8.29.0 \
    ipykernel~=6.29.0 \
    jupyterlab~=4.2.0 \
  '';
}
