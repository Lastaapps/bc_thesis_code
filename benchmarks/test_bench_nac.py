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
    "subgraphs-False-random",
    "subgraphs-True-degree",
    "subgraphs-True-degree_cycles",
    "subgraphs-True-cycles",
    "subgraphs-True-cycles_match_chunks",
    "subgraphs-True-components_biggest",
    "subgraphs-True-components_spredded",
    "subgraphs-False-none",
    "subgraphs-False-degree",
    "subgraphs-False-degree_cycles",
    "subgraphs-False-cycles",
    "subgraphs-False-cycles_match_chunks",
    "subgraphs-False-components_biggest",
    "subgraphs-False-components_spredded",
]
NAC_ALGORITHMS_FAST = [
    "subgraphs-True-none",
    "subgraphs-True-random",
    "subgraphs-True-degree",
    "subgraphs-True-degree_cycles",
    "subgraphs-True-cycles",
    "subgraphs-True-cycles_match_chunks",
]
NAC_ALGORITHMS_I_AM_SPEED = [
    "subgraphs-True-none",
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
@pytest.mark.parametrize("dataset", [laman_medium_graphs[:32]], ids=["laman_medium"])
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
@pytest.mark.parametrize("bridges", [True])
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS_I_AM_SPEED)
@pytest.mark.parametrize(
    ("vertices_no", "graph_cnt", "first_n"),
    [
        (16, 128, 1024),
        (17, 128, 1024),
        (18, 128, 1024),
        (19, 128, 1024),
        (20, 128, 1024),
        (21, 128, 512),
        (22, 128, 512),
        (23, 128, 512),
        (24, 64, 512),
        (25, 64, 512),
        (26, 64, 256),
        (27, 64, 256),
        (28, 64, 128),
        (29, 64, 128),
        (30, 64, 128),
    ],
)
def test_bench_NAC_colorings_laman_large_first_n(
    benchmark,
    algorithm: str,
    bridges: bool,
    vertices_no: int,
    graph_cnt: int,
    first_n: int,
):
    """
    Measures the time to find first 32 NAC colorings of the graph given if they
    exist. The reason for this test is that you don't usually need all the NAC
    colorings of a graph and some algorithms may use it to their advantage.
    """

    dataset = list(
        filter(lambda g: nx.number_of_nodes(g) == vertices_no, laman_large_graphs)
    )[:graph_cnt]

    def perform_test():
        for i, graph in enumerate(dataset):
            # print("Graph:   ", i, graph_cnt)
            j = -1
            for j, _ in zip(
                range(first_n),
                graph.NAC_colorings(
                    algorithm=algorithm,
                    use_bridges_decomposition=bridges,
                ),
            ):
                pass
                # if j == 0:
                #     print("Coloring:", j, graph_cnt)
            # print("Coloring:", j, graph_cnt)

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
