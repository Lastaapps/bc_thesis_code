from typing import List
from benchmarks.dataset import load_all_graphs
from pyrigi.graph import Graph
import pytest

SMALL_GRAPH_FILE_LIMIT = 256
SMALL_GRAPH_CNT = 128
SMALL_GRAPH_FUZZY_LIMIT = 2048
small_graphs: List[Graph] = load_all_graphs(SMALL_GRAPH_FILE_LIMIT)


@pytest.mark.parametrize("algorithm", ["naive", "cycles", "subgraphs"])
def test_bench_single_NAC_colorings(benchmark, algorithm: str):
    def perform_test():
        for graph in small_graphs[:256]:
            graph.single_NAC_coloring(algorithm=algorithm)

    benchmark(perform_test)


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


@pytest.mark.parametrize("algorithm", ["cycles", "subgraphs"])
def test_NAC_coloring_small_graphs(algorithm: str):
    for graph in small_graphs[:SMALL_GRAPH_FUZZY_LIMIT]:
        print(graph)

        naive = graph.NAC_colorings(algorithm="naive")
        tested = graph.NAC_colorings(algorithm=algorithm)

        naive = [(tuple(coloring[0]), tuple(coloring[1])) for coloring in naive]
        tested = [(tuple(coloring[0]), tuple(coloring[1])) for coloring in tested]

        assert set(naive) == set(tested)
