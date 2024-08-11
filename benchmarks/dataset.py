from collections import defaultdict
import itertools
import os
import random
import re
import sys
import urllib.request
from enum import Enum
import networkx as nx
import zipfile
from typing import Dict, Iterable, List, Literal

from pyrigi.graph import Graph


class GraphFamily(Enum):
    REDUNDANTLY_RIGID = (7473079, "RedundantlyRigidGraphs")
    GLOBALLY_RIGID = (7473053, "GloballyRigidGraphs")
    LAMAN = (1245517, "LamanGraphs")


STORE_DIR = "./benchmarks/graphs_store"
LAMAN_DIR = f"{STORE_DIR}/nauty-laman/general"
LAMAN_DEGREE_3_PLUS_DIR = f"{STORE_DIR}/nauty-laman/degree_3_plus"
GENERAL_DIR = f"{STORE_DIR}/general-graphs"


def download_small_graphs(family: GraphFamily, size: str) -> None:
    url = "https://zenodo.org/records/{}/files/{}{}.zip?download=1".format(
        family.value[0], family.value[1], size
    )

    name = family.value[1] + " " + size
    path = "{}/{}.zip".format(STORE_DIR, name)
    os.makedirs(STORE_DIR, exist_ok=True)
    if not os.path.exists(path):
        print("Downloading {} dataset...".format(name), file=sys.stderr)
        urllib.request.urlretrieve(url, filename="{}.tmp".format(path))
        os.rename("{}.tmp".format(path), path)

        print("Extracting {} dataset...".format(name), file=sys.stderr)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(STORE_DIR, name))


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
    dir = os.path.join(STORE_DIR, name)

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


def _filter_triangle_only_laman_graphs(graphs) -> filter:
    return filter(
        lambda g: len(Graph._find_triangle_components(g)[1]) > 1,
        graphs,
    )


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


def load_laman_degree_3_plus(shuffle: bool = True) -> Iterable[Graph]:
    return load_laman_graphs(dir = LAMAN_DEGREE_3_PLUS_DIR, shuffle=shuffle)


LAMAN_DEGREE_3_PLUS_ALL_DIR = f"{STORE_DIR}/laman_degree_3_plus"
LAMAN_DEGREE_3_PLUS_ALL_FILENAME = "D3LamanGraphs{}.m"

def load_laman_degree_3_plus_all(
    vertices_no: int, limit: int | None = None
) -> Iterable[Graph]:
    path = os.path.join(
        LAMAN_DEGREE_3_PLUS_ALL_DIR, LAMAN_DEGREE_3_PLUS_ALL_FILENAME.format(vertices_no)
    )

    graph_no = 0
    if limit == None:
        limit = -1  # Will never end

    with open(path) as file:
        for line in file:
            for integer in re.finditer("(\\d+)", line):
                yield Graph.from_int(int(integer.group()))

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


def load_general_graphs(
    category: Literal["simple", "hard"],
    dir: str = GENERAL_DIR,
) -> Dict[str, List[Graph]]:
    limit: int | None = 64

    prefix_to_name = {
        "cubhypo": "hypohamiltonian_cubic",
        "highlyirregular": "highly_irregular",
        "hypo": "hypohamiltonian",
        "selfcomp": "self_complementary",
        "planar_conn": "planar",
        "c34_": "no_3,4_cycles",
    }

    dir = os.path.join(dir, category)

    graphs: Dict[str, List[Graph]] = defaultdict(list)
    for file in os.listdir(dir):
        name = None
        for prefix in prefix_to_name:
            if file.startswith(prefix):
                name = prefix_to_name[prefix]
                break

        path = os.path.join(dir, file)

        if file.endswith(".g6"):
            assert name
            # print(f"Loading file {path}")

            if limit is None:
                graphs[name] += [Graph(g) for g in nx.read_graph6(path)]
            else:
                with open(path, mode="rb") as input_file:
                    for _, line in zip(range(limit), input_file):
                        line = line.strip()
                        if not len(line):
                            continue
                        graphs[name] += [Graph(nx.from_graph6_bytes(line))]
        elif file.endswith(".s6"):
            assert name
            # print(f"Loading file {path}")

            if limit is None:
                graphs[name] += [Graph(g) for g in nx.read_graph6(path)]
            else:
                with open(path, mode="rb") as input_file:
                    for _, line in zip(range(limit), input_file):
                        line = line.strip()
                        if not len(line):
                            continue
                        graphs[name] += [Graph(nx.from_sparse6_bytes(line))]

    for key in graphs.keys():
        graphs[key] = [Graph(g) for g in graphs[key]]

    return graphs


# kneser_graph(n, k)[source]
#
# Returns the Kneser Graph with parameters n and k.
# The Kneser Graph has nodes that are k-tuples (subsets) of the integers between 0 and n-1. Nodes are adjacent if their corresponding sets are disjoint.
#
# Parameters:
#     n: int
#         Number of integers from which to make node subsets. Subsets are drawn from set(range(n)).
#     k: int
#         Size of the subsets.


def load_small_generated_graphs(limit: int = 128) -> Dict[str, List[Graph]]:
    graphs: Dict[str, List[Graph]] = {}

    rand = random.Random(42)
    randint = rand.randint

    # fmt: off
    # print(f"Generating graphs")
    # average (#, #) (vertices,edges)/graph

    graphs["sparse_small"] = [nx.gnm_random_graph(randint(10, 14), randint(17, 22), randint(0, 1234)) for _ in range(limit)]

    graphs["dense_small"] = [nx.gnm_random_graph(randint(7, 9), randint(20, 25), randint(0, 1234)) for _ in range(limit)]
    # fmt: on

    for key in graphs.keys():
        data = [Graph(g) for g in graphs[key]]
        rand.shuffle(data)
        graphs[key] = data

    return graphs


def load_medium_generated_graphs(limit: int = 128) -> Dict[str, List[Graph]]:
    graphs: Dict[str, List[Graph]] = {}

    rand = random.Random(42)
    randint = rand.randint

    # fmt: off
    # print(f"Generating graphs")
    # average (#, #) (vertices,edges)/graph

    # Kneser graphs have pairs as vertices
    graphs["kneser"] = [nx.kneser_graph(randint(5, 6), randint(2, 3)) for _ in range(limit)] # (13.7060546875, 17.1533203125)

    graphs["sparse_medium"] = [nx.gnm_random_graph(randint(14, 17), randint(22, 27), randint(0, 1234)) for _ in range(limit)]

    graphs["dense_medium"] = [nx.gnm_random_graph(randint(14, 17), randint(28, 34), randint(0, 1234)) for _ in range(limit)]
    # fmt: on

    for key in graphs.keys():
        data = [Graph(g) for g in graphs[key]]
        rand.shuffle(data)
        graphs[key] = data

    return graphs


def load_generated_graphs(limit: int = 128) -> Dict[str, List[Graph]]:
    graphs: Dict[str, List[Graph]] = {}

    rand = random.Random(42)
    randint = rand.randint

    # fmt: off
    # print(f"Generating graphs")
    # average (#, #) (vertices,edges)/graph

    # I decided they don't make sense to check anyway, as they are known class of graphs in terms of NAC coloring
    # graphs["complete_bipartite"] = [nx.complete_multipartite_graph(*[randint(2, 4) for _ in range(3)]) for _ in range(limit)]
    # graphs["complete_tripartite"] = [nx.complete_multipartite_graph(*[randint(2, 4) for _ in range(3)]) for _ in range(limit)]

    # Kneser graphs have pairs as vertices
    # graphs["kneser"] = [nx.kneser_graph(randint(3, 5), randint(2, 3)) for _ in range(limit)] # (5.5849609375, 3.03515625)
    graphs["kneser"] = [nx.kneser_graph(randint(5, 6), randint(2, 3)) for _ in range(limit)] # (13.7060546875, 17.1533203125)
    # graphs["kneser"] = [nx.kneser_graph(randint(5, 7), randint(2, 4)) for _ in range(limit)] # (18.7685546875, 26.923828125)
    # graphs["kneser"] = [nx.kneser_graph(randint(6, 8), randint(2, 6)) for _ in range(limit)] # (27.4609375, 49.228515625)

    # graphs["random_sparse"] = [nx.gnm_random_graph(randint(10, 30), randint(16, 28), randint(0, 1234)) for _ in range(limit)]
    graphs["random_sparse_medium"] = [nx.gnm_random_graph(randint(20, 25), randint(30, 40), randint(0, 1234)) for _ in range(limit)]
    # graphs["random_sparse_large"] = [nx.gnm_random_graph(randint(26, 32), randint(40, 50), randint(0, 1234)) for _ in range(limit)]

    graphs["random_dense_medium"] = [nx.gnm_random_graph(randint(15, 20), randint(40, 50), randint(0, 1234)) for _ in range(limit)]

    # graphs["random_regular"] = [nx.random_regular_graph(randint(4, 6)//2*2, randint(12, 18), randint(0, 1234)) for _ in range(limit)] # (15.0498046875, 35.4189453125)
    graphs["random_regular"] = (
        [nx.random_regular_graph(4, randint(14, 16), randint(0, 1234)) for _ in range(limit // 2)] + # (15.0068359375, 30.013671875)
        [nx.random_regular_graph(6, randint(10, 12), randint(0, 1234)) for _ in range(limit // 2)] # (10.9814453125, 32.9443359375)
    )
    # fmt: on

    for key in graphs.keys():
        data = [Graph(g) for g in graphs[key]]
        rand.shuffle(data)
        graphs[key] = data

    return graphs
