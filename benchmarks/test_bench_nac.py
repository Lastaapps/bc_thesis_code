from typing import List
from benchmarks.dataset import (
    load_all_small_graphs,
    load_general_graphs,
    load_generated_graphs,
    load_laman_graphs,
)
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
laman_larger_graphs = list(
    filter(lambda g: nx.number_of_nodes(g) in range(16, 17 + 1), laman_graphs)
)
laman_large_graphs = list(filter(lambda g: nx.number_of_nodes(g) > 15, laman_graphs))

general_graphs = load_general_graphs()
generated_graphs = load_generated_graphs()

# can be used for debugging
# small_graphs = sorted(small_graphs, key=lambda g: nx.number_of_nodes(g))

NAC_ALGORITHMS = [
    "naive",
    "cycles-True",
    "cycles-False",
    "subgraphs-True-none-auto",
    "subgraphs-False-random-auto",
    "subgraphs-True-degree-auto",
    "subgraphs-True-degree_cycles-auto",
    "subgraphs-True-cycles-auto",
    "subgraphs-True-cycles_match_chunks-auto",
    "subgraphs-True-components_biggest-auto",
    "subgraphs-True-components_spredded-auto",
    "subgraphs-False-none-auto",
    "subgraphs-False-degree-auto",
    "subgraphs-False-degree_cycles-auto",
    "subgraphs-False-cycles-auto",
    "subgraphs-False-cycles_match_chunks-auto",
    "subgraphs-False-components_biggest-auto",
    "subgraphs-False-components_spredded-auto",
]
NAC_ALGORITHMS_FAST = [
    "subgraphs-True-none-auto",
    "subgraphs-True-random-auto",
    "subgraphs-True-degree-auto",
    "subgraphs-True-degree_cycles-auto",
    "subgraphs-True-cycles-auto",
    "subgraphs-True-cycles_match_chunks-auto",
]
NAC_RELABEL_STRATEGIES = [
    "none",
    "random",
    "bfs",
    "beam-degree",
]
# pairs (algorithm, relabel)
NAC_FASTEST = [
    ("subgraphs-True-none-auto", "none"),
    ("subgraphs-True-beam_neighbors-4", "random"),
    ("subgraphs-True-beam_neighbors-5", "random"),
    ("subgraphs-True-beam_neighbors-6", "random"),
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
@pytest.mark.parametrize("bridges", [True])
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS_FAST)
@pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES)
@pytest.mark.parametrize(
    "dataset",
    [laman_medium_graphs[:32], laman_larger_graphs[:16]],
    ids=["laman_medium", "laman_larger"],
)
def test_bench_NAC_colorings_fast(
    benchmark,
    algorithm: str,
    relabel_strategy: str,
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
                relabel_strategy=relabel_strategy,
                use_bridges_decomposition=bridges,
            ):
                pass

    benchmark(perform_test)


@pytest.mark.nac_benchmark
@pytest.mark.parametrize(("algorithm", "relabel_strategy"), NAC_FASTEST)
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
    relabel_strategy: str,
    vertices_no: int,
    graph_cnt: int,
    first_n: int,
    bridges: bool = True,
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
                    relabel_strategy=relabel_strategy,
                    use_bridges_decomposition=bridges,
                ),
            ):
                pass
                # if j == 0:
                #     print("Coloring:", j, graph_cnt)
            # print("Coloring:", j, graph_cnt)

    benchmark(perform_test)


@pytest.mark.nac_benchmark
@pytest.mark.parametrize(("algorithm", "relabel_strategy"), NAC_FASTEST)
@pytest.mark.parametrize(
    ("dataset", "first_n"),
    [(general_graphs[:8], 512), (generated_graphs[:8], 512)],
    ids=["general", "generated"],
)
def test_bench_NAC_colorings_general_first_n(
    benchmark,
    dataset: List[Graph],
    algorithm: str,
    relabel_strategy: str,
    first_n: int,
    bridges: bool = True,
):
    """
    Measures the time to find first 32 NAC colorings of the graph given if they
    exist. The reason for this test is that you don't usually need all the NAC
    colorings of a graph and some algorithms may use it to their advantage.
    """

    def perform_test():
        for i, graph in enumerate(dataset):
            print("Graph:   ", i, graph)
            j = -1
            for j, _ in zip(
                range(first_n),
                graph.NAC_colorings(
                    algorithm=algorithm,
                    relabel_strategy=relabel_strategy,
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

    naive = {Graph.canonical_NAC_coloring(coloring) for coloring in naive}
    tested = {Graph.canonical_NAC_coloring(coloring) for coloring in tested}

    s1, s2 = len(naive), len(tested)
    assert s1 == s2
    assert l1 == s1
    assert l2 == s2
    assert naive == tested


def test_wtf_is_going_on():
    import time

    # limit = 1_000_000
    # limit = 10_000
    limit = 1_000

    # dataset: List[Graph] = list(
    #     filter(lambda g: nx.number_of_nodes(g) == 15, laman_medium_graphs)
    # )[:30][
    #     15:16
    # ]  # [7:8]
    # dataset = general_graphs[:10]
    dataset = general_graphs[6:7]

    for graph in dataset:
        results = []

        def test(
            algorithm: str = "subgraphs",
            relabel_strategy: str = "none",
            use_bridges_decomposition: bool = True,
        ):
            print(f"Test {algorithm} {relabel_strategy}")
            start = time.time()
            data = []
            for i, c in zip(
                range(limit),
                graph.NAC_colorings(
                    algorithm=algorithm,
                    relabel_strategy=relabel_strategy,
                    use_bridges_decomposition=use_bridges_decomposition,
                ),
            ):
                data.append(c)
                if i % 100 == 0:
                    print(i)
            # print(len(data))
            duration = time.time() - start
            results.append(duration)
            print(f"Duration: {duration}")

        # test(
        #     algorithm="subgraphs-True-none-auto",
        #     relabel_strategy="none",
        # )
        # for i in [7, 13]: # 42 edges
        #     test(
        #         algorithm=f"subgraphs-True-none-{i}",
        #         relabel_strategy="none",
        #     )
        #######################################################################

        if True:
            # 11
            test(
                algorithm="subgraphs-True-beam_neighbors_smart-4",
                relabel_strategy="none",
            )
            test(
                algorithm="subgraphs-True-beam_neighbors_smart-8",
                relabel_strategy="none",
            )
            test(
                algorithm="subgraphs-True-beam_neighbors_smart-4",
                relabel_strategy="random",
            )
            test(
                algorithm="subgraphs-True-beam_neighbors_smart-8",
                relabel_strategy="random",
            )

        if False:
            # 43
            test(
                algorithm="subgraphs-True-beam_neighbors-4",
                relabel_strategy="random",
            )
            # 12
            test(
                algorithm="subgraphs-True-beam_neighbors-4",
                relabel_strategy="bfs",
            )
            # 30
            test(
                algorithm="subgraphs-True-beam_neighbors-4",
                relabel_strategy="beam_degree",
            )

            # trash
            test(
                algorithm="subgraphs-True-beam_neighbors-5",
                relabel_strategy="none",
            )
            test(
                algorithm="subgraphs-True-beam_neighbors-5",
                relabel_strategy="random",
            )
            test(
                algorithm="subgraphs-True-beam_neighbors-5",
                relabel_strategy="bfs",
            )
            test(
                algorithm="subgraphs-True-beam_neighbors-5",
                relabel_strategy="beam_degree",
            )

        print(results)
        # print()
        # print()
        # print()
        # print()
        # print()
        # print()
