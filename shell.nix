# TODO convert to flake
{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  name = "PyRigi shell";

  packages = [
    (pkgs.python312.withPackages (
      python-pkgs: with python-pkgs; [
        pandas
        numpy
        matplotlib
        scipy
        sympy
        networkx
        pytest
        pytest-benchmark
        pytest-timeout
        tqdm
        urllib3

        black

        # notebook
        # ipython
        # jupyter
      ]
    ))
    (pkgs.python311.withPackages (
      python-pkgs: with python-pkgs; [
        # these do not support python 3.12 yet
        notebook
        ipython
        jupyter

        pandas
        numpy
        matplotlib
        scipy
        sympy
        networkx
        pytest
        pytest-benchmark
        pytest-timeout
        tqdm
        urllib3
      ]
    ))
    pkgs.jupyter-all
  ];

  shellHook = '''';
}
