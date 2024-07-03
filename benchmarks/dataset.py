import itertools
import os
import random
import sys
import urllib.request
from enum import Enum
import networkx as nx
import zipfile
from typing import List

from pyrigi.graph import Graph


class GraphFamily(Enum):
    REDUNDANTLY_RIGID = (7473079, "RedundantlyRigidGraphs")
    GLOBALLY_RIGID = (7473053, "GloballyRigidGraphs")
    LAMAN = (1245517, "LamanGraphs")


DOWNLOAD_DIR = "./benchmarks/graphs_store"
LAMAN_DIR = "./benchmarks/graphs_store/nauty-laman"


def download_small_graphs(family: GraphFamily, size: str) -> None:
    url = "https://zenodo.org/records/{}/files/{}{}.zip?download=1".format(
        family.value[0], family.value[1], size
    )

    name = family.value[1] + " " + size
    path = "{}/{}.zip".format(DOWNLOAD_DIR, name)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    if not os.path.exists(path):
        print("Downloading {} dataset...".format(name), file=sys.stderr)
        urllib.request.urlretrieve(url, filename="{}.tmp".format(path))
        os.rename("{}.tmp".format(path), path)

        print("Extracting {} dataset...".format(name), file=sys.stderr)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(DOWNLOAD_DIR, name))


configurations = [
    (GraphFamily.REDUNDANTLY_RIGID, "D1"),
    (GraphFamily.REDUNDANTLY_RIGID, "D2"),
    (GraphFamily.REDUNDANTLY_RIGID, "D3"),
    (GraphFamily.GLOBALLY_RIGID, "D1"),
    (GraphFamily.GLOBALLY_RIGID, "D2"),
    (GraphFamily.GLOBALLY_RIGID, "D3"),
    # (GraphFamily.LAMAN, "3-10"),
    # (GraphFamily.LAMAN, "11"),
    # (GraphFamily.LAMAN, "12"),
]


def download_small_all_graphs():
    for family, type in configurations:
        download_small_graphs(family, type)


def load_small_graph(family: GraphFamily, size: str, limit: int | None):
    download_small_graphs(family, size)

    name = family.value[1] + " " + size
    dir = os.path.join(DOWNLOAD_DIR, name)

    graphs: List[Graph] = []
    for file in os.listdir(dir):
        if not file.endswith(".g6"):
            continue

        path = os.path.join(dir, file)
        # print(f"Loading file {path}")

        if limit is None:
            graphs += [Graph(g) for g in nx.read_graph6(path)]
        else:
            with open(path, mode="rb") as input_file:
                for _, line in zip(range(limit), input_file):
                    line = line.strip()
                    if not len(line):
                        continue
                    graphs.append(Graph(nx.from_graph6_bytes(line)))

    return graphs


def load_all_small_graphs(limit: int | None, shuffle: bool = True) -> List[Graph]:
    graphs: List[Graph] = []
    for family, type in configurations:
        graphs += load_small_graph(family, type, limit)

    if shuffle:
        random.Random(42).shuffle(graphs)

    return graphs

def load_laman_graphs(dir: str= LAMAN_DIR, shuffle: bool = True):

    graphs: List[Graph] = []
    for file in os.listdir(dir):
        if not file.endswith(".g6"):
            continue

        path = os.path.join(dir, file)
        # print(f"Loading file {path}")

        graphs += [Graph(g) for g in nx.read_graph6(path)]

    if shuffle:
        random.Random(42).shuffle(graphs)

    return graphs
