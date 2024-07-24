import random
from typing import List
from benchmarks.dataset import (
    load_all_small_graphs,
    load_general_graphs,
    load_generated_graphs,
    load_laman_graphs,
    load_medium_generated_graphs,
    load_small_generated_graphs,
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

general_small_graphs = load_small_generated_graphs()
general_small_graphs = general_small_graphs.items()

general_medium_graphs = load_general_graphs("simple") | load_medium_generated_graphs()
general_medium_graphs = general_medium_graphs.items()

general_graphs = load_general_graphs("hard") | load_generated_graphs()
general_quick = ["complete_multipartite", "self_complementary", "highly_irregular"]
general_disabled = ["kneser"]  # + general_quick
for key in general_disabled:
    general_graphs.pop(key, None)
general_graphs = general_graphs.items()


NAC_ALGORITHMS = [
    "naive",
    "cycles-True",
    "cycles-False",
    "subgraphs-True-none-4",
    "subgraphs-False-random-4",
    "subgraphs-True-degree-4",
    "subgraphs-True-degree_cycles-4",
    "subgraphs-True-cycles-4",
    "subgraphs-True-cycles_match_chunks-4",
    "subgraphs-True-components_biggest-4",
    "subgraphs-True-components_spredded-4",
    "subgraphs-False-none-4",
    "subgraphs-False-degree-4",
    "subgraphs-False-degree_cycles-4",
    "subgraphs-False-cycles-4",
    "subgraphs-False-cycles_match_chunks-4",
    "subgraphs-False-components_biggest-4",
    "subgraphs-False-components_spredded-4",
    "subgraphs-True-bfs-4",
    "subgraphs-True-beam_neighbors-4",
    "subgraphs-True-beam_neighbors_triangles-4",
    "subgraphs-True-beam_neighbors_max-4",
    "subgraphs-True-beam_neighbors_max_triangles-4",
    "subgraphs-True-none-4-smart",
    "subgraphs-False-random-4-smart",
    "subgraphs-True-degree-4-smart",
    "subgraphs-True-degree_cycles-4-smart",
    "subgraphs-True-cycles-4-smart",
    "subgraphs-True-cycles_match_chunks-4-smart",
    "subgraphs-True-components_biggest-4-smart",
    "subgraphs-True-components_spredded-4-smart",
    "subgraphs-False-none-4-smart",
    "subgraphs-False-degree-4-smart",
    "subgraphs-False-degree_cycles-4-smart",
    "subgraphs-False-cycles-4-smart",
    "subgraphs-False-cycles_match_chunks-4-smart",
    "subgraphs-False-components_biggest-4-smart",
    "subgraphs-False-components_spredded-4-smart",
    "subgraphs-True-bfs-4-smart",
    "subgraphs-True-beam_neighbors-4-smart",
    "subgraphs-True-beam_neighbors_triangles-4-smart",
    "subgraphs-True-beam_neighbors_max-4-smart",
    "subgraphs-True-beam_neighbors_max_triangles-4-smart",
]
NAC_RELABEL_STRATEGIES = [
    "none",
    "random",
    "bfs",
    "beam_degree",
]


@pytest.mark.nac_benchmark
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
@pytest.mark.parametrize(
    "dataset",
    [small_graphs[:32], laman_small_graphs[:32]]
    + [v[:32] for _, v in general_small_graphs],
    ids=["small", "laman_small"] + [k for k, _ in general_small_graphs],
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

    benchmark.pedantic(perform_test, warmup_rounds=0)


BENCH_ROUNDS_SMALL = 3


@pytest.mark.timeout(60 * BENCH_ROUNDS_SMALL)
@pytest.mark.nac_benchmark
# @pytest.mark.parametrize("bridges", [True, False])
@pytest.mark.parametrize("bridges", [True])
@pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES)
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
@pytest.mark.parametrize(
    "dataset",
    [small_graphs[:128], laman_small_graphs[:64]]
    + [v[:32] for _, v in general_small_graphs],
    ids=["small", "laman_small"] + [k for k, _ in general_small_graphs],
)
def test_bench_NAC_colorings_small(
    benchmark,
    algorithm: str,
    bridges: bool,
    relabel_strategy: str,
    dataset: List[Graph],
):
    """
    Measures the time to find all the NAC colorings of the graph given if any
    exists. This can also get slow really quickly for some algorithms.
    """
    rand = random.Random(42)

    def perform_test():

        for graph in dataset:
            for _ in graph.NAC_colorings(
                algorithm=algorithm,
                use_bridges_decomposition=bridges,
                use_has_coloring_check=False,
                relabel_strategy=relabel_strategy,
                seed=rand.randint(0, 2**32 - 1),
            ):
                pass

    benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_SMALL, warmup_rounds=0)


NAC_ALGORITHMS_LAMAN_FAST = [
    "subgraphs-True-none-8-smart",
    "subgraphs-True-degree-8-smart",
    "subgraphs-True-degree_cycles-8-smart",
    "subgraphs-True-cycles-8-smart",
    "subgraphs-True-cycles_match_chunks-8-smart",
    "subgraphs-True-beam_neighbors-8-smart",
    "subgraphs-True-beam_neighbors_triangles-8-smart",
    "subgraphs-True-beam_neighbors_max-8-smart",
    "subgraphs-True-beam_neighbors_max_triangles-8-smart",
]
NAC_RELABEL_STRATEGIES_LAMAN_FAST = [
    "none",
    "random",
    "bfs",
    "beam_degree",
]


BENCH_ROUNDS_LAMAN_FAST = 5


@pytest.mark.nac_benchmark
@pytest.mark.timeout(240 * BENCH_ROUNDS_LAMAN_FAST)
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS_LAMAN_FAST)
@pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES_LAMAN_FAST)
@pytest.mark.parametrize(
    "dataset",
    [
        # laman_medium_graphs[:32],
        laman_larger_graphs[:16],
    ],
    ids=[
        # "laman_medium",
        "laman_larger",
    ],
)
def test_bench_NAC_colorings_laman_fast(
    benchmark,
    algorithm: str,
    relabel_strategy: str,
    dataset: List[Graph],
):
    """
    Measures the time to find all the NAC colorings of the graph given if any
    exists. This can also get slow really quickly for some algorithms.
    """
    rand = random.Random(42)
    print()

    def perform_test():
        from tqdm import tqdm
        for graph in tqdm(dataset):
        # for graph in dataset:
            for _ in graph.NAC_colorings(
                algorithm=algorithm,
                relabel_strategy=relabel_strategy,
                use_bridges_decomposition=True,
                use_has_coloring_check=False,
                seed=rand.randint(0, 2**32 - 1),
            ):
                pass

    benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_LAMAN_FAST, warmup_rounds=0)


NAC_ALGORITHMS_GENERAL_FAST = [
    "subgraphs-True-none-4-smart",
    "subgraphs-True-degree-4-smart",
    "subgraphs-True-degree_cycles-4-smart",
    "subgraphs-True-cycles-4-smart",
    "subgraphs-True-cycles_match_chunks-4-smart",
    "subgraphs-True-beam_neighbors-4-smart",
    "subgraphs-True-beam_neighbors_triangles-4-smart",
    "subgraphs-True-beam_neighbors_max-4-smart",
    "subgraphs-True-beam_neighbors_max_triangles-4-smart",
]
NAC_RELABEL_STRATEGIES_GENERAL_FAST = [
    # "none",
    # "random",
    # "bfs",
    "beam_degree",
]


BENCH_ROUNDS_GENERAL_MEDIUM = 3


@pytest.mark.nac_benchmark
@pytest.mark.timeout(180 * BENCH_ROUNDS_GENERAL_MEDIUM)
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS_GENERAL_FAST)
@pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES_GENERAL_FAST)
@pytest.mark.parametrize(
    "dataset",
    [v for _, v in general_medium_graphs],
    ids=[k for k, _ in general_medium_graphs],
)
@pytest.mark.parametrize("graph_cnt", [32])
def test_bench_NAC_colorings_general_medium(
    benchmark,
    algorithm: str,
    relabel_strategy: str,
    dataset: List[Graph],
    graph_cnt: int,
):
    """
    Measures the time to find all the NAC colorings of the graph given if any
    exists. This can also get slow really quickly for some algorithms.
    """
    rand = random.Random(42)
    # print()
    dataset = dataset[:graph_cnt]

    def perform_test():
        # from tqdm import tqdm
        # for graph in tqdm(dataset):

        for graph in dataset:
            for _ in graph.NAC_colorings(
                algorithm=algorithm,
                relabel_strategy=relabel_strategy,
                use_bridges_decomposition=True,
                use_has_coloring_check=False,
                seed=rand.randint(0, 2**32 - 1),
            ):
                pass

    benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_GENERAL_MEDIUM)


# pairs (algorithm, relabel)
NAC_FASTEST_LAMAN = [
    # smart is not working well here
    ("subgraphs-True-none-4-smart", "none"),
    ("subgraphs-True-beam_neighbors-4-smart", "beam_degree"),
    # ("subgraphs-True-none-4", "none"),
    # ("subgraphs-True-none-8", "none"),
    # ("subgraphs-True-none-4", "random"),
    # ("subgraphs-True-none-8", "random"),
    # ("subgraphs-True-degree_cycles-8", "random"),
    # ("subgraphs-True-cycles-8", "random"),
    # ("subgraphs-True-cycles_match_chunks-8", "random"),
    # ("subgraphs-True-beam_neighbors-4", "random"),
    # ("subgraphs-True-beam_neighbors-8", "random"),
    # ("subgraphs-True-beam_neighbors_triangles-4", "random"),
    # ("subgraphs-True-beam_neighbors_triangles-8", "random"),
    # ("subgraphs-True-beam_neighbors_max-4", "random"),
    # ("subgraphs-True-beam_neighbors_max-8", "random"),
    # ("subgraphs-True-beam_neighbors_max_triangles-4", "random"),
    # ("subgraphs-True-beam_neighbors_max_triangles-8", "random"),
]

BENCH_ROUNDS_LAMAN_LARGE = 4


@pytest.mark.nac_benchmark
@pytest.mark.timeout(360 * BENCH_ROUNDS_LAMAN_LARGE)
@pytest.mark.parametrize(("algorithm", "relabel_strategy"), NAC_FASTEST_LAMAN)
@pytest.mark.parametrize(
    ("vertices_no", "graph_cnt", "first_n"),
    [
        (16, 64, 1024),
        (17, 64, 1024),
        (18, 64, 1024),
        (19, 64, 1024),
        (20, 64, 1024),
        (21, 64, 512),
        (22, 64, 512),
        (23, 64, 512),
        (24, 32, 512),
        (25, 32, 512),
        (26, 32, 256),
        (27, 32, 256),
        (28, 32, 128),
        (29, 32, 128),
        (30, 32, 128),
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
    rand = random.Random(42)
    print()

    dataset = list(
        filter(lambda g: nx.number_of_nodes(g) == vertices_no, laman_large_graphs)
    )[:graph_cnt]

    def perform_test():

        from tqdm import tqdm

        for graph in tqdm(dataset):
            j = -1
            for j, _ in zip(
                range(first_n),
                graph.NAC_colorings(
                    algorithm=algorithm,
                    relabel_strategy=relabel_strategy,
                    use_bridges_decomposition=bridges,
                    use_has_coloring_check=False,
                    seed=rand.randint(0, 2**32 - 1),
                ),
            ):
                pass
                # if j == 0:
                #     print("Coloring:", j, graph_cnt)
            # print("Coloring:", j, graph_cnt)

    benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_LAMAN_LARGE, warmup_rounds=0)


NAC_FASTEST_GENERAL = [
    ("subgraphs-True-none-4", "none"),
    ("subgraphs-True-none-8", "none"),
    ("subgraphs-True-degree-4", "random"),
    ("subgraphs-True-degree-8", "random"),
    ("subgraphs-True-degree_cycles-4", "random"),
    ("subgraphs-True-degree_cycles-8", "random"),
    ("subgraphs-True-cycles-4", "random"),
    ("subgraphs-True-cycles-8", "random"),
    ("subgraphs-True-cycles_match_chunks-4", "random"),
    ("subgraphs-True-cycles_match_chunks-8", "random"),
    ("subgraphs-True-beam_neighbors-4", "random"),
    ("subgraphs-True-beam_neighbors-8", "random"),
    ("subgraphs-True-beam_neighbors_max-4", "random"),
    ("subgraphs-True-beam_neighbors_max-8", "random"),
    ("subgraphs-True-beam_neighbors_triangles-4", "random"),
    ("subgraphs-True-beam_neighbors_triangles-8", "random"),
    ("subgraphs-True-beam_neighbors_max_triangles-4", "random"),
    ("subgraphs-True-beam_neighbors_max_triangles-8", "random"),
    # ("subgraphs-True-none-4-smart", "none"),
    # ("subgraphs-True-none-8-smart", "none"),
    # ("subgraphs-True-degree-4-smart", "random"),
    # ("subgraphs-True-degree-8-smart", "random"),
    # ("subgraphs-True-degree_cycles-4-smart", "random"),
    # ("subgraphs-True-degree_cycles-8-smart", "random"),
    # ("subgraphs-True-cycles-4-smart", "random"),
    # ("subgraphs-True-cycles-8-smart", "random"),
    # ("subgraphs-True-cycles_match_chunks-4-smart", "random"),
    # ("subgraphs-True-cycles_match_chunks-8-smart", "random"),
    # ("subgraphs-True-bfs-4-smart", "random"),
    # ("subgraphs-True-bfs-8-smart", "random"),
    # ("subgraphs-True-beam_neighbors-4-smart", "random"),
    # ("subgraphs-True-beam_neighbors-8-smart", "random"),
    # ("subgraphs-True-beam_neighbors_max-4-smart", "random"),
    # ("subgraphs-True-beam_neighbors_max-8-smart", "random"),
    # ("subgraphs-True-beam_neighbors_triangles-4-smart", "random"),
    # ("subgraphs-True-beam_neighbors_triangles-8-smart", "random"),
    # ("subgraphs-True-beam_neighbors_max_triangles-4-smart", "random"),
    # ("subgraphs-True-beam_neighbors_max_triangles-8-smart", "random"),
]


BENCH_ROUNDS_GENERAL_LARGE = 3


@pytest.mark.nac_benchmark
@pytest.mark.timeout(180 * BENCH_ROUNDS_GENERAL_LARGE)
@pytest.mark.parametrize(("algorithm", "relabel_strategy"), NAC_FASTEST_GENERAL)
@pytest.mark.parametrize(
    "dataset",
    [v[:32] for _, v in general_graphs],
    ids=[k for k, _ in general_graphs],
)
@pytest.mark.parametrize("first_n", [1, 512])
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
    rand = random.Random(42)
    print()

    def perform_test():
        from tqdm import tqdm

        for graph in tqdm(dataset):
            j = None
            for j, _ in zip(
                range(first_n),
                graph.NAC_colorings(
                    algorithm=algorithm,
                    relabel_strategy=relabel_strategy,
                    use_bridges_decomposition=bridges,
                    use_has_coloring_check=False,
                    seed=rand.randint(0, 2**32 - 1),
                ),
            ):
                pass
                # if j == 0:
                #     print("Coloring:", j)
            # print(f"Colorings:{j}\n")

    benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_GENERAL_LARGE, warmup_rounds=0)


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
@pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES)
@pytest.mark.parametrize("graph", small_graphs[:SMALL_GRAPH_FUZZY_LIMIT])
def test_NAC_coloring_small_graphs(algorithm: str, graph: Graph, relabel_strategy: str):
    """
    Checks algorithm validity against the naive implementation
    (that is hopefully correct) and checks that outputs are the same.
    Large number of smaller graphs is used (naive implementation is really slow for larger ones).
    """

    # print(graph)
    naive = list(graph.NAC_colorings(algorithm="naive"))
    tested = list(
        graph.NAC_colorings(algorithm=algorithm, relabel_strategy=relabel_strategy)
    )

    l1, l2 = len(naive), len(tested)
    assert l1 == l2

    # for coloring in naive:
    #     graph.is_NAC_coloring(coloring)
    # for coloring in tested:
    #     graph.is_NAC_coloring(coloring)

    naive = {
        Graph.canonical_NAC_coloring(coloring, including_red_blue_order=False)
        for coloring in naive
    }
    tested = {
        Graph.canonical_NAC_coloring(coloring, including_red_blue_order=False)
        for coloring in tested
    }

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
    # dataset = general_graphs[6:7]
    dataset = [general_graphs[i] for i in [6, 10, 13]]

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
                    use_has_coloring_check=False,
                    seed=rand.randint(0, 2**32 - 1),
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
            # test(
            #     algorithm="subgraphs-True-bfs-4",
            #     relabel_strategy="none",
            # )
            # test(
            #     algorithm="subgraphs-True-bfs-8",
            #     relabel_strategy="none",
            # )
            # test(
            #     algorithm="subgraphs-True-bfs_smart-4",
            #     relabel_strategy="none",
            # )
            # test(
            #     algorithm="subgraphs-True-bfs_smart-8",
            #     relabel_strategy="none",
            # )
            test(
                algorithm="subgraphs-True-beam_neighbors_smart_triangles-4",
                relabel_strategy="none",
            )
            # test(
            #     algorithm="subgraphs-True-beam_neighbors_smart_triangles-8",
            #     relabel_strategy="none",
            # )

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
