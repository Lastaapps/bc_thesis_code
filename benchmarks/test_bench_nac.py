from typing import List
from benchmarks.dataset import load_all_small_graphs, load_laman_graphs
from pyrigi.graph import Graph
import pytest
import networkx as nx

SMALL_GRAPH_FILE_LIMIT = 64
SMALL_GRAPH_FUZZY_LIMIT = 256
small_graphs: List[Graph] = load_all_small_graphs(SMALL_GRAPH_FILE_LIMIT)
laman_graphs: List[Graph] = load_laman_graphs()
laman_small_graphs = list(filter(lambda g: nx.number_of_nodes(g) < 10, laman_graphs))
laman_medium_graphs = list(
    filter(lambda g: nx.number_of_nodes(g) in range(10, 15 + 1), laman_graphs)
)
laman_large_graphs = list(filter(lambda g: nx.number_of_nodes(g) > 15, laman_graphs))

# can be used for debugging
# small_graphs = sorted(small_graphs, key=lambda g: nx.number_of_nodes(g))

NAC_ALGORITHMS = [
    "naive",
    "cycles-True",
    "cycles-False",
    "subgraphs-True-none",
    "subgraphs-True-rank",
    "subgraphs-True-rank_cycles",
    "subgraphs-True-cycles",
    "subgraphs-True-cycles_match_chunks",
    "subgraphs-True-components_biggest",
    "subgraphs-True-components_spredded",
    "subgraphs-False-none",
    "subgraphs-False-rank",
    "subgraphs-False-rank_cycles",
    "subgraphs-False-cycles",
    "subgraphs-False-cycles_match_chunks",
    "subgraphs-False-components_biggest",
    "subgraphs-False-components_spredded",
]
NAC_ALGORITHMS_FAST = [
    "subgraphs-True-none",
    "subgraphs-True-rank",
    "subgraphs-True-rank_cycles",
    "subgraphs-True-cycles",
    "subgraphs-True-cycles_match_chunks",
]


@pytest.mark.nac_benchmark
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
@pytest.mark.parametrize(
    "dataset",
    [small_graphs[:32], laman_small_graphs[:32]],
    ids=["small", "laman_small"],
)
def test_bench_single_NAC_colorings(
    benchmark,
    algorithm: str,
    dataset: List[Graph],
):
    """
    Measures the time till the first NAC coloring of the graph given is found.
    (or it is decided there is no NAC coloring).
    """

    def perform_test():
        for graph in dataset:
            graph.single_NAC_coloring(algorithm=algorithm)

    benchmark(perform_test)


@pytest.mark.nac_benchmark
@pytest.mark.parametrize("bridges", [True, False])
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
@pytest.mark.parametrize(
    "dataset",
    [small_graphs[:32], laman_small_graphs[:32]],
    ids=["small", "laman_small"],
)
def test_bench_NAC_colorings(
    benchmark,
    algorithm: str,
    bridges: bool,
    dataset: List[Graph],
):
    """
    Measures the time to find all the NAC colorings of the graph given if any
    exists. This can also get slow really quickly for some algorithms.
    """

    def perform_test():
        for graph in dataset:
            for _ in graph.NAC_colorings(
                algorithm=algorithm,
                use_bridges_decomposition=bridges,
            ):
                pass

    benchmark(perform_test)


@pytest.mark.nac_benchmark
@pytest.mark.parametrize("bridges", [True, False])
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS_FAST)
@pytest.mark.parametrize(
    "dataset",
    [laman_medium_graphs[:32], laman_large_graphs[:8]],
    ids=["laman_medium", "laman_large"],
)
def test_bench_NAC_colorings_fast(
    benchmark,
    algorithm: str,
    bridges: bool,
    dataset: List[Graph],
):
    """
    Measures the time to find all the NAC colorings of the graph given if any
    exists. This can also get slow really quickly for some algorithms.
    """

    def perform_test():
        for graph in dataset:
            for _ in graph.NAC_colorings(
                algorithm=algorithm,
                use_bridges_decomposition=bridges,
            ):
                pass

    benchmark(perform_test)


@pytest.mark.nac_benchmark
@pytest.mark.parametrize("bridges", [True, False])
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
@pytest.mark.parametrize(
    "dataset",
    [small_graphs[:32], laman_small_graphs[:32]],
    ids=["small", "laman_small"],
)
def test_bench_NAC_colorings_first_32(
    benchmark,
    algorithm: str,
    bridges: bool,
    dataset: List[Graph],
    first_n: int = 32,
):
    """
    Measures the time to find first 32 NAC colorings of the graph given if they
    exist. The reason for this test is that you don't usually need all the NAC
    colorings of a graph and some algorithms may use it to their advantage.
    """

    def perform_test():
        for graph in dataset:
            for _ in zip(
                range(first_n),
                graph.NAC_colorings(
                    algorithm=algorithm,
                    use_bridges_decomposition=bridges,
                ),
            ):
                pass

    benchmark(perform_test)


@pytest.mark.parametrize("algorithm", ["cycles", "subgraphs"])
@pytest.mark.parametrize("graph", small_graphs[:SMALL_GRAPH_FUZZY_LIMIT])
def test_NAC_coloring_small_graphs(algorithm: str, graph: Graph):
    """
    Checks algorithm validity against the naive implementation
    (that is hopefully correct) and checks that outputs are the same.
    Large number of smaller graphs is used (naive implementation is really slow for larger ones).
    """

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
