"""
This modules is used to generate and store random graphs of the given class.
There graphs can later be loaded using the dataset module
"""

import os
import random
import math
from typing import *
from pathlib import Path

import networkx as nx
from tqdm import tqdm

import networkx as nx
import nac
from nac.util import NiceGraph as Graph
from benchmarks import dataset

# TODO import PyRigi

################################################################################


class RangeWithCount(NamedTuple):
    """
    Represents range of values with assigned count
    The high bound is excluded
    """

    low: int
    high: int
    cnt: int


def _write_graphs_to_file(
    path: str | Path,
    graphs: Sequence[nx.Graph],
):
    with open(path, "wb") as f:
        for graph in graphs:
            f.write(nx.readwrite.graph6.to_graph6_bytes(graph, header=False))


def _generate_random_graphs_impl(
    ranges_with_count: Sequence[RangeWithCount],
    generate_graph: Callable[[int, int | None], nx.Graph],
    dir: str,
    filename_template: str,
    seed: int | None,
) -> List[Tuple[int, List[nx.Graph]]]:
    """
    Base function for generating the graphs
    """
    os.makedirs(dir, exist_ok=True)
    rand = random.Random(seed)

    configs = [(n, c) for l, h, c in ranges_with_count for n in range(l, h)]
    results: List[Tuple[int, List[nx.Graph]]] = []
    for n, count in tqdm(configs):
        path = Path(os.path.join(dir, f"{filename_template.format(n)}.g6"))

        if path.is_file():
            graphs = dataset.load_graph6_graphs_from_file(str(path))
        else:
            graphs = []

        # skip already generated graphs
        for _ in range(len(graphs)):
            rand.randint(0, 2**30)

        graphs += [
            generate_graph(n, rand.randint(0, 2**30))
            for _ in range(count - len(graphs))
        ]

        _write_graphs_to_file(path, graphs)

        results.append((n, graphs))
    return results


# takes ~1h 30m on my laptop
def generate_random_laman_graphs(
    dir: str = dataset.LAMAN_DIR_RANDOM,
    filename_template: str = "laman_{0}",
    seed: int | None = 42,
) -> List[Tuple[int, List[nx.Graph]]]:

    ranges = (
        RangeWithCount(10, 20, 128),
        RangeWithCount(20, 30, 64),
        RangeWithCount(30, 40, 32),
        RangeWithCount(40, 50, 16),
        RangeWithCount(50, 60, 8),
    )
    return _generate_random_graphs_impl(
        ranges,
        lambda n, seed: _generate_laman_graph(n, seed),
        dir,
        filename_template,
        seed,
    )


def generate_random_globally_rigid_graphs(
    dir: str = os.path.join(dataset.RANDOM_DIR, "globally_rigid"),
    filename_template: str = "globally_rigid_{0}",
    seed: int | None = 42,
) -> List[Tuple[int, List[nx.Graph]]]:

    ranges = (
        RangeWithCount(10, 20, 100),
        RangeWithCount(20, 30, 100),
        RangeWithCount(30, 40, 100),
        RangeWithCount(40, 50, 100),
        RangeWithCount(50, 60, 100),
    )
    return _generate_random_graphs_impl(
        ranges,
        lambda n, seed: _generate_globally_rigid_graph(n, seed),
        dir,
        filename_template,
        seed,
    )


def generate_random_sparse_with_few_colorings_graphs(
    dir: str = os.path.join(dataset.RANDOM_DIR, "sparse_with_few_colorings"),
    filename_template: str = "sparse_with_few_colorings_{0}",
    seed: int | None = 42,
) -> List[Tuple[int, List[nx.Graph]]]:

    ranges = (
        RangeWithCount(10, 20, 100),
        RangeWithCount(20, 30, 100),
        RangeWithCount(30, 40, 100),
        RangeWithCount(40, 50, 100),
        RangeWithCount(50, 60, 100),
    )
    return _generate_random_graphs_impl(
        ranges,
        lambda n, seed: _generate_NAC_critical_graph(n, seed),
        dir,
        filename_template,
        seed,
    )


################################################################################
# Generate a single graph of the given class
################################################################################


def _generate_laman_graph(
    n: int,
    seed: int | None,
    min_degree: int | None = None,
) -> nx.Graph:
    import pyrigi.graph

    rand = random.Random(seed)

    while True:
        graph = pyrigi.Graph(nx.gnm_random_graph(n, 2 * n - 3, rand.randint(0, 2**30)))
        if min_degree is not None:
            if next((1 for d in nx.degree(graph) if d < min_degree), None) is not None:
                continue
        if not nx.is_connected(graph):
            continue
        if not graph.is_min_rigid():
            continue
        if len(nac.find_monochromatic_classes(graph)[1]) == 1:
            continue

        return graph


def _generate_sparse_graph(
    n: int,
    seed: int | None,
    p: float = 0.1,
) -> Graph:
    rand = random.Random(seed)
    while True:
        graph = Graph(nx.fast_gnp_random_graph(n, p, seed=rand.randint(0, 2**30)))
        if not nx.is_connected(graph):
            continue

        return graph


# does not make sense as dense graphs form a single monochromatic class
def _generate_dense_graph(
    n: int,
    seed: int | None,
    p: float = 0.8,
) -> Graph:
    rand = random.Random(seed)
    while True:
        graph = Graph(nx.gnp_random_graph(n, p, seed))
        if not nx.is_connected(graph):
            continue

        return graph


def _generate_NAC_critical_graph(
    n: int,
    seed: int | None,
    log_base: float = math.e,
) -> nx.Graph:
    """
    Generates sparse graphs that should have likely few NAC colorings
    or no NAC colorings what so ever
    """
    rand = random.Random(seed)

    # this formula comes from a related
    # and still unpublished work of related authors
    # it should lead to graphs that either have
    # no NAC-coloring or have just few

    p = 0.95 * (2 * math.log(n, log_base) / (n * n)) ** (1 / 3)

    while True:
        graph = Graph(nx.fast_gnp_random_graph(n, p, seed=rand.randint(0, 2**30)))
        if not nx.is_connected(graph):
            continue

        return graph


def _generate_globally_rigid_graph(
    n: int,
    seed: int | None,
    log_base: float = math.e,
) -> nx.Graph:
    """
    Generates sparse graphs that should have likely few NAC colorings
    or no NAC colorings what so ever
    """
    import pyrigi

    rand = random.Random(seed)

    # this formula comes from a related
    # and still unpublished work of related authors
    # it should lead to graphs that either have
    # no NAC-coloring or have just few

    p = (2 * math.log(n, log_base) / (n * n)) ** (1 / 3)
    while True:
        graph = pyrigi.Graph(
            nx.fast_gnp_random_graph(n, p, seed=rand.randint(0, 2**30))
        )
        if not nx.is_connected(graph):
            continue
        if len(nac.find_monochromatic_classes(graph)[1]) == 1:
            continue
        if not graph.is_globally_rigid():
            continue

        return graph
