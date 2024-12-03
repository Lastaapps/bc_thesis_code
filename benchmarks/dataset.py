"""
This module is used for loading and generating graphs for benchmarks
Currently the middle part is not used.
The top part processes Laman and Redundantly-Rigid Graphs.
The middle part generates and loads some not as interesting graph classes
The bottom part generates sparse graphs of the given size.
"""

import os
import random
import re
import networkx as nx
from typing import *

import networkx as nx
import nac
from nac.util import NiceGraph as Graph

STORE_DIR = os.path.join("graphs_store")
RANDOM_DIR = os.path.join(STORE_DIR, "random")
LAMAN_DIR_NAUTY = os.path.join(STORE_DIR, "nauty")
LAMAN_DIR = os.path.join(LAMAN_DIR_NAUTY, "laman_some")
LAMAN_DIR_DEGREE_3_PLUS = os.path.join(LAMAN_DIR_NAUTY, "laman_some_degree_3_plus")
LAMAN_DIR_RANDOM = os.path.join(RANDOM_DIR, "laman")
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


def load_graph6_graphs_from_dir(
    dir: str,
) -> List[Graph]:
    graphs: List[Graph] = []

    for file in os.listdir(dir):
        path = os.path.join(dir, file)

        graphs += load_graph6_graphs_from_file(path)

    return graphs


def load_graph6_graphs_from_file(
    path: str,
) -> List[Graph]:
    graphs: List[Graph] = []

    if path.endswith(".g6"):
        # print(f"Loading file {path}")

        graphs.extend(nx.read_graph6(path))
    if path.endswith(".s6"):
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
    return load_graph6_graphs_from_dir(os.path.join(STORE_DIR, "no_3_nor_4_cycles"))


def load_globally_rigid_graphs() -> List[Graph]:
    return load_graph6_graphs_from_dir(os.path.join(RANDOM_DIR, "globally_rigid"))


def load_sparse_with_few_colorings_graphs() -> List[Graph]:
    return load_graph6_graphs_from_dir(
        os.path.join(RANDOM_DIR, "sparse_with_few_colorings")
    )
