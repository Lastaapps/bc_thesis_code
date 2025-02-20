# Flexible realizations existence: NP-completeness on sparse graphs and algorithms

*Petr Laštovička and Jan Legerský*

In this repository we provide our code for the NAC-coloring search algorithm
described in the paper. We also provide a notebook `NAC_playground.ipynb` where
you can experiment with the algorithm and `NAC_presentation.ipynb`
in which you can see how we run and analyze our benchmarks. You can also run them yourself.
The interface of our code is also described in the notebooks, mainly in `NAC_presentation.ipynb`.

## Setup

Python 3.12 is required.

```bash
pip -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

You can run tests by executing `pytest`.
The packages also contains base for Cartesian NAC-coloring search,
the related tests are skipped for now as it is not yet fully implemented
for every approach.

## Structure
- `nac` - the code of our NAC implementation
- `benchmarks` - Core related to benchmarks - graphs loading, generation, notebook utility functions
- `benchmarks/precomputed` - Results of the benchmarks as run on our hardware
- `graphs_store` - stores generated graphs of selected classes
