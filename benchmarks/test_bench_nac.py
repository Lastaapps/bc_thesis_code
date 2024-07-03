from typing import List
from benchmarks.dataset import load_all_graphs
from pyrigi.graph import Graph
import networkx as nx
import pytest

SMALL_GRAPH_FILE_LIMIT = 64
SMALL_GRAPH_CNT = 128
SMALL_GRAPH_FUZZY_LIMIT = 256
small_graphs: List[Graph] = load_all_graphs(SMALL_GRAPH_FILE_LIMIT)

# can be used for debugging
# small_graphs_from_smallest = sorted(small_graphs, key=lambda g: nx.number_of_nodes(g))


@pytest.mark.parametrize("algorithm", ["naive", "cycles", "subgraphs"])
def test_bench_single_NAC_colorings(benchmark, algorithm: str):
    def perform_test():
        for graph in small_graphs[:256]:
            graph.single_NAC_coloring(algorithm=algorithm)

    benchmark(perform_test)


# TODO add subgraph order strategies
@pytest.mark.parametrize("bridges", [True, False])
@pytest.mark.parametrize("algorithm", ["naive", "cycles", "subgraphs"])
def test_bench_NAC_colorings(benchmark, algorithm: str, bridges: bool):
    def perform_test():
        for graph in small_graphs[:32]:
            for _ in graph.NAC_colorings(
                algorithm=algorithm,
                use_bridges_decomposition=bridges,
            ):
                pass

    benchmark(perform_test)


# TODO add subgraph order strategies
@pytest.mark.parametrize("algorithm", ["cycles", "subgraphs"])
@pytest.mark.parametrize("graph", small_graphs[:SMALL_GRAPH_FUZZY_LIMIT])
def test_NAC_coloring_small_graphs(algorithm: str, graph: Graph):
    # print(graph)
    naive = list(graph.NAC_colorings(algorithm="naive"))
    tested = list(graph.NAC_colorings(algorithm=algorithm))

    l1, l2 = len(naive), len(tested)
    assert l1 == l2

    # for coloring in naive:
    #     graph.is_NAC_coloring(coloring)
    # for coloring in tested:
    #     graph.is_NAC_coloring(coloring)

    naive = {
        (tuple(sorted(coloring[0])), tuple(sorted(coloring[1]))) for coloring in naive
    }
    tested = {
        (tuple(sorted(coloring[0])), tuple(sorted(coloring[1]))) for coloring in tested
    }

    s1, s2 = len(naive), len(tested)
    assert s1 == s2
    assert l1 == s1
    assert l2 == s2
    assert naive == tested
