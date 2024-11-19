# distinctipy.nix
{
  buildPythonPackage,
  fetchPypi,
  setuptools,
  wheel,
}:

buildPythonPackage rec {
  pname = "distinctipy";
  version = "1.3.4";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-/tl6//Gvtz7KqHyFRhAh8LqJ+uYwZ8ASW5ZzUmUQqsQ=";
  };

  doCheck = false;

  pyproject = true;
  build-system = [
    setuptools
    wheel
  ];
}

