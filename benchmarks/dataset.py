"""
This module is used for loading and generating graphs for benchmarks
Currently the middle part is not used.
The top part processes Laman and Redundantly-Rigid Graphs.
The middle part generates and loads some not as interesting graph classes
The bottom part generates sparse graphs of the given size.
"""

from collections import defaultdict
import os
import random
import re
import sys
import math
import urllib.request
from enum import Enum
import networkx as nx
import zipfile
from typing import *

import networkx as nx
import nac
from nac.util import NiceGraph as Graph

STORE_DIR = os.path.join("graphs_store")
LAMAN_DIR_NAUTY = os.path.join(STORE_DIR, "nauty")
LAMAN_DIR = os.path.join(LAMAN_DIR_NAUTY, "laman_some")
LAMAN_DIR_DEGREE_3_PLUS = os.path.join(LAMAN_DIR_NAUTY, "laman_some_degree_3_plus")
LAMAN_DIR_RANDOM = os.path.join(STORE_DIR, "random", "laman")
GENERAL_DIR = os.path.join(STORE_DIR, "general-graphs")


################################################################################
def _filter_triangle_only_laman_graphs(graphs) -> filter:
    return filter(
        lambda g: len(nac.find_monochromatic_classes(g)[1]) > 1,
        graphs,
    )

def _filter_non_connected(graphs: Iterable[Graph]) -> Iterable[Graph]:
    for graph in graphs:
        if nx.is_connected(graph):
            yield graph
################################################################################


def load_laman_graphs(dir: str = LAMAN_DIR, shuffle: bool = True) -> Iterable[Graph]:
    graphs: List[Graph] = []
    for file in os.listdir(dir):
        if not file.endswith(".g6"):
            continue

        path = os.path.join(dir, file)
        # print(f"Loading file {path}")

        graphs += [Graph(g) for g in nx.read_graph6(path)]

    if shuffle:
        random.Random(42).shuffle(graphs)

    return _filter_triangle_only_laman_graphs(graphs)


def load_laman_random_graphs(shuffle: bool = True) -> Iterable[Graph]:
    return load_laman_graphs(dir=LAMAN_DIR_RANDOM, shuffle=shuffle)


def load_laman_degree_3_plus(shuffle: bool = True) -> Iterable[Graph]:
    return load_laman_graphs(dir=LAMAN_DIR_DEGREE_3_PLUS, shuffle=shuffle)


LAMAN_ALL_DIR = os.path.join(LAMAN_DIR_NAUTY, "laman_all")
LAMAN_ALL_FILENAME = "laman_{}.g6"
LAMAN_DEGREE_3_PLUS_ALL_DIR = os.path.join(STORE_DIR, "laman_degree_3_plus")
LAMAN_DEGREE_3_PLUS_ALL_FILENAME = "D3LamanGraphs{}.m"


def load_laman_all(
    vertices_no: int,
    DIR: str = LAMAN_ALL_DIR,
    FILENAME: str | None = None,
) -> Iterable[nx.Graph]:
    if FILENAME is None:
        FILENAME = LAMAN_ALL_FILENAME.format(vertices_no)
    path = os.path.join(DIR, FILENAME)
    return nx.read_graph6(path)


def load_laman_degree_3_plus_all(
    vertices_no: int, limit: int | None = None
) -> Iterable[nx.Graph]:
    import pyrigi

    path = os.path.join(
        LAMAN_DEGREE_3_PLUS_ALL_DIR,
        LAMAN_DEGREE_3_PLUS_ALL_FILENAME.format(vertices_no),
    )

    graph_no = 0
    if limit == None:
        limit = -1  # Will never end

    with open(path) as file:
        for line in file:
            for integer in re.finditer("(\\d+)", line):
                yield pyrigi.Graph.from_int(int(integer.group()))

                graph_no += 1
                if graph_no == limit:
                    return


def _convert_g6_to_int(from_path: str, to_path: str):
    """
    This function was used for the routine above to commonize sources
    """
    with open(from_path, mode="rb") as input_file, open(
        to_path, mode="w"
    ) as output_file:
        output_file.write("{")
        is_first = True
        for line in input_file:
            line = line.strip()
            if not len(line):
                continue
            graph = Graph(nx.from_graph6_bytes(line))
            if not is_first:
                output_file.write(",\n")
                is_first = False
            output_file.write(str(graph.to_int()))
        output_file.write("}")

def load_graph6_graphs(
    dir: str,
) -> List[Graph]:
    graphs: List[Graph] = []

    for file in os.listdir(dir):
        path = os.path.join(dir, file)

        if file.endswith(".g6"):
            # print(f"Loading file {path}")

            graphs.extend(nx.read_graph6(path))
        if file.endswith(".g6") or file.endswith(".s6"):
            # print(f"Loading file {path}")

            with open(path, mode="rb") as input_file:
                for line in input_file:
                    line = line.strip()
                    if not len(line):
                        continue
                    g = nx.from_sparse6_bytes(line.__bytes__())
                    graphs.append(Graph(g))

    return graphs

def load_no_3_nor_4_cycle_graphs() -> List[Graph]:
    return load_graph6_graphs(os.path.join(STORE_DIR, "no_3_nor_4_cycles"))

################################################################################
# Random graphs generation and search
################################################################################


def generate_laman_graphs(
    nodes_l: int,
    nodes_h: int,
    count: int = 64,
    min_degree: int | None = None,
    seed: int | None = 42,
) -> List[nx.Graph]:
    graphs: List[nx.Graph] = list()
    rand = random.Random(seed)
    import pyrigi.graph

    for n in range(nodes_l, nodes_h + 1):
        found = 0
        while found < count:
            graph = pyrigi.Graph(
                nx.gnm_random_graph(n, 2 * n - 3, rand.randint(0, 2**32))
            )
            if min_degree is not None:
                if (
                    next((1 for d in nx.degree(graph) if d < min_degree), None)
                    is not None
                ):
                    continue
            if not nx.is_connected(graph):
                continue
            if not graph.is_min_rigid():
                continue
            if len(nac.find_monochromatic_classes(graph)[1]) == 1:
                continue

            graphs.append(graph)
            found += 1
    rand.shuffle(graphs)
    return graphs


def generate_sparse_graphs(
    nodes_l: int,
    nodes_h: int,
    count: int = 64,
    seed: int | None = 42,
    p: float = 0.1,
) -> List[Graph]:
    graphs: List[Graph] = list()
    rand = random.Random(seed)

    for n in range(nodes_l, nodes_h + 1):
        for _ in range(count):
            graphs.append(Graph(nx.fast_gnp_random_graph(n, p, rand.randint(0, 2**32))))
    graphs = list(_filter_non_connected(graphs))
    rand.shuffle(graphs)
    return graphs


# does not make sense as dense graphs form a single monochromatic class
def generate_dense_graphs(
    nodes_l: int,
    nodes_h: int,
    count: int = 64,
    seed: int | None = 42,
    p: float = 0.8,
) -> List[Graph]:
    graphs: List[Graph] = list()
    rand = random.Random(seed)

    for n in range(nodes_l, nodes_h + 1):
        for _ in range(count):
            graphs.append(Graph(nx.gnp_random_graph(n, p, rand.randint(0, 2**32))))
    graphs = list(_filter_non_connected(graphs))
    rand.shuffle(graphs)
    return graphs


def generate_NAC_critical_graphs(
    nodes_l: int,
    nodes_h: int,
    count: int | None = None,
    seed: int | None = 42,
    log_base: float = math.e,
) -> Iterable[nx.Graph]:
    """
    Generates sparse graphs that should have likely few NAC colorings
    or no NAC colorings what so ever
    """
    rand = random.Random(seed)

    # this formula comes from a related
    # and still unpublished work of related authors
    # it should lead to graphs that either have
    # no NAC-coloring or have just few

    i = 0
    while i < (count or 2**30):
        n = rand.randint(nodes_l, nodes_h)
        p = 0.95 * (2 * math.log(n, log_base) / (n * n)) ** (1 / 3)

        while True:
            graph = Graph(nx.fast_gnp_random_graph(n, p, rand.randint(0, 2**32)))
            if not nx.is_connected(graph):
                continue

            i += 1
            yield graph
            break


def generate_globally_rigid_graphs(
    nodes_l: int,
    nodes_h: int,
    count: int | None = None,
    seed: int | None = 42,
    log_base: float = math.e,
) -> Iterable[nx.Graph]:
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

    i = 0
    while i < (count or 2**30):
        n = rand.randint(nodes_l, nodes_h)
        p = (2 * math.log(n, log_base) / (n * n)) ** (1 / 3)

        while True:
            graph = pyrigi.Graph(nx.fast_gnp_random_graph(n, p, rand.randint(0, 2**32)))
            if not nx.is_connected(graph):
                continue
            if len(nac.find_monochromatic_classes(graph)[1]) == 1:
                continue
            if not graph.is_globally_rigid():
                continue

            i += 1
            yield graph
            break
