from collections import defaultdict, deque
import numpy as np
import itertools
import random
from typing import Container, Dict, Iterable, List
from benchmarks.dataset import (
    load_all_small_graphs,
    load_general_graphs,
    load_generated_graphs,
    load_laman_degree_3_plus,
    load_laman_degree_3_plus_all,
    load_laman_graphs,
    load_medium_generated_graphs,
    load_small_generated_graphs,
)
from pyrigi.graph import Graph, print_stats
import pytest
import networkx as nx


SMALL_GRAPH_FILE_LIMIT = 64
SMALL_GRAPH_FUZZY_LIMIT = 256
small_graphs: List[Graph] = load_all_small_graphs(SMALL_GRAPH_FILE_LIMIT)
laman_graphs: List[Graph] = list(load_laman_graphs())
laman_degree_3_plus_graphs: List[Graph] = list(load_laman_degree_3_plus())
laman_small_graphs = list(filter(lambda g: nx.number_of_nodes(g) < 10, laman_graphs))
laman_medium_graphs = list(
    filter(lambda g: nx.number_of_nodes(g) in range(10, 15 + 1), laman_graphs)
)
laman_larger_graphs = list(
    filter(lambda g: nx.number_of_nodes(g) in range(16, 17 + 1), laman_graphs)
)
laman_large_graphs = list(filter(lambda g: nx.number_of_nodes(g) > 15, laman_graphs))
laman_medium_degree_3_plus_graphs = list(
    filter(
        lambda g: nx.number_of_nodes(g) in range(12, 15 + 1), laman_degree_3_plus_graphs
    )
)


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


################################################################################
NAC_ALGORITHMS = [
    "naive",
    "cycles-True",
    "cycles-False",
    "subgraphs-linear-none-4",
    "subgraphs-linear-cycles-4",
    "subgraphs-linear-cycles_match_chunks-4",
    "subgraphs-linear-neighbors-4",
    "subgraphs-linear-neighbors_cycles-4",
    "subgraphs-linear-neighbors_degree-4",
    "subgraphs-linear-neighbors_iterative-4",
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
    [small_graphs[:64], laman_small_graphs[:64]]
    + [v[:64] for _, v in general_small_graphs],
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


################################################################################
BENCH_ROUNDS_SMALL = 3


# @pytest.mark.timeout(60 * BENCH_ROUNDS_SMALL)
# @pytest.mark.nac_benchmark
# @pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
# @pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES)
# @pytest.mark.parametrize(
#     "dataset",
#     [small_graphs[:128], laman_small_graphs[:64]]
#     + [v[:64] for _, v in general_small_graphs],
#     ids=["small", "laman_small"] + [k for k, _ in general_small_graphs],
# )
# def test_bench_NAC_colorings_small(
#     benchmark,
#     algorithm: str,
#     relabel_strategy: str,
#     dataset: List[Graph],
# ):
#     """
#     Measures the time to find all the NAC colorings of the graph given if any
#     exists. This can also get slow really quickly for some algorithms.
#     """
#     rand = random.Random(42)

#     def perform_test():

#         for graph in dataset:
#             for _ in graph.NAC_colorings(
#                 algorithm=algorithm,
#                 use_decompositions=True,
#                 use_has_coloring_check=False,
#                 relabel_strategy=relabel_strategy,
#                 seed=rand.randint(0, 2**32 - 1),
#             ):
#                 pass

#     benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_SMALL, warmup_rounds=0)


@pytest.mark.timeout(60 * BENCH_ROUNDS_SMALL)
@pytest.mark.nac_benchmark
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
@pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES)
@pytest.mark.parametrize(
    "dataset",
    [laman_medium_graphs[:64]],
    ids=["laman_medium"],
)
def test_bench_NAC_colorings_small(
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

    def perform_test():

        for graph in dataset:
            for _ in graph.NAC_colorings(
                algorithm=algorithm,
                use_decompositions=True,
                use_has_coloring_check=False,
                relabel_strategy=relabel_strategy,
                seed=rand.randint(0, 2**32 - 1),
            ):
                pass

    benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_SMALL, warmup_rounds=0)


################################################################################
NAC_ALGORITHMS_LAMAN_FAST = [
    "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
    for merge in [
        "linear",
        "log",
        # "log_reverse",
        "sorted_bits",
        # "sorted_size",
        "score",
        "recursion",
    ]
    for algo in [
        "none",
        # "cycles",
        "cycles_match_chunks",
        "neighbors",
        "neighbors_cycle",
        "neighbors_degree",
        # "neighbors_degree_cycle",
        # "neighbors_iterative",
    ]
    for size in [4, 6, 8]
]
NAC_RELABEL_STRATEGIES_LAMAN_FAST = [
    "none",
    "random",
    "bfs",
    # "beam_degree",
]


BENCH_ROUNDS_LAMAN_FAST = 3


@pytest.mark.nac_benchmark
@pytest.mark.timeout(20 * BENCH_ROUNDS_LAMAN_FAST)
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS_LAMAN_FAST)
@pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES_LAMAN_FAST)
@pytest.mark.parametrize(
    "dataset",
    [
        laman_medium_graphs[:64],
        laman_larger_graphs[:32],
        laman_medium_degree_3_plus_graphs[:32],
    ],
    ids=[
        "laman_medium",
        "laman_larger",
        "laman_medium_degree_3_plus_graphs",
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
                use_decompositions=True,
                use_has_coloring_check=False,
                seed=rand.randint(0, 2**32 - 1),
            ):
                pass

    benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_LAMAN_FAST, warmup_rounds=0)


################################################################################
NAC_ALGORITHMS_GENERAL_FAST = [
    "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
    for merge in [
        "linear",
        "log",
        # "log_reverse",
        "sorted_bits",
        # "sorted_size",
        "score",
        "recursion",
    ]
    for algo in [
        "none",
        # "cycles",
        "cycles_match_chunks",
        "neighbors",
        "neighbors_cycle",
        "neighbors_degree",
        # "neighbors_degree_cycle",
        # "neighbors_iterative",
    ]
    for size in [4, 6, 8]
] + []

NAC_RELABEL_STRATEGIES_GENERAL_FAST = [
    # "none",
    "random",
    "bfs",
    # "beam_degree",
]


BENCH_ROUNDS_GENERAL_MEDIUM = 3


@pytest.mark.nac_benchmark
@pytest.mark.timeout(120 * BENCH_ROUNDS_GENERAL_MEDIUM)
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
            # print(graph)
            for _ in graph.NAC_colorings(
                algorithm=algorithm,
                relabel_strategy=relabel_strategy,
                use_decompositions=True,
                use_has_coloring_check=False,
                seed=rand.randint(0, 2**32 - 1),
            ):
                pass

    benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_GENERAL_MEDIUM)


################################################################################
# pairs (algorithm, relabel)
NAC_FASTEST_LAMAN = [
    ("random", "subgraphs-{}-{}-{}-smart".format(merge, algo, size))
    for merge in [
        "linear",
        "log",
        "sorted_bits",
    ]
    for algo in [
        # "cycles",
        "cycles_match_chunks",
        "neighbors",
        "neighbors_cycle",
        "neighbors_degree",
        # "neighbors_degree_cycle",
        # "neighbors_iterative",
    ]
    for size in [6, 8]
] + [
    ("subgraphs-linear-none-6", "none"),
    ("subgraphs-linear-none-8", "none"),
]

BENCH_ROUNDS_LAMAN_LARGE = 3


@pytest.mark.nac_benchmark
@pytest.mark.timeout(15 * BENCH_ROUNDS_LAMAN_LARGE)
@pytest.mark.parametrize(("relabel_strategy", "algorithm"), NAC_FASTEST_LAMAN)
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
    use_decompositions: bool = True,
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
                    use_decompositions=use_decompositions,
                    use_has_coloring_check=False,
                    seed=rand.randint(0, 2**32 - 1),
                ),
            ):
                pass
                # if j == 0:
                #     print("Coloring:", j, graph_cnt)
            # print("Coloring:", j, graph_cnt)

    benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_LAMAN_LARGE, warmup_rounds=0)


################################################################################
NAC_FASTEST_GENERAL = [
    ("random", "subgraphs-{}-{}-{}-smart".format(merge, algo, size))
    for merge in [
        "linear",
        "log",
        "log_reverse",
        "sorted_bits",
    ]
    for algo in [
        # "cycles",
        "cycles_match_chunks",
        "neighbors",
        "neighbors_cycle",
        "neighbors_degree",
        # "neighbors_degree_cycle",
        # "neighbors_iterative",
    ]
    for size in [6, 8]
] + []


BENCH_ROUNDS_GENERAL_LARGE = 3


@pytest.mark.nac_benchmark
@pytest.mark.timeout(120 * BENCH_ROUNDS_GENERAL_LARGE)
@pytest.mark.parametrize(("relabel_strategy", "algorithm"), NAC_FASTEST_GENERAL)
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
    use_decompositions: bool = True,
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
                    use_decompositions=use_decompositions,
                    use_has_coloring_check=False,
                    seed=rand.randint(0, 2**32 - 1),
                ),
            ):
                pass
                # if j == 0:
                #     print("Coloring:", j)
            # print(f"Colorings:{j}\n")

    benchmark.pedantic(perform_test, rounds=BENCH_ROUNDS_GENERAL_LARGE, warmup_rounds=0)


################################################################################
NAC_LAMAN_DEG_3_PLUS = [
    "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
    for merge in [
        "linear",
        "log",
        "log_reverse",
        "sorted_bits",
        "sorted_size",
        "score",
        "recursion",
    ]
    for algo in [
        "cycles_match_chunks",
        "neighbors",
        "neighbors_cycle",
        "neighbors_degree",
        "neighbors_degree_cycle",
        "neighbors_iterative",
    ]
    for size in [6, 8]
] + []
NAC_LAMAN_DEG_3_PLUS = ["subgraphs-log-neighbors-6-smart"]


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", NAC_LAMAN_DEG_3_PLUS)
@pytest.mark.parametrize(("vertex_no", "rounds"), [(9, 5), (10, 3)])
# @pytest.mark.parametrize(("vertex_no", "rounds"), [(9, 1), (10, 1), (11, 1), (12, 1)])
def test_NAC_coloring_laman_degree_3_plus(
    benchmark, algorithm: str, vertex_no: int, rounds: int
):
    from tqdm import tqdm
    from pandas import DataFrame

    stats = defaultdict(int)
    rand = random.Random(42)
    dataset = list(load_laman_degree_3_plus_all(vertex_no))

    print()

    def perform_test():
        for graph in tqdm(dataset):
            iterable = graph.NAC_colorings(
                algorithm=algorithm,
                relabel_strategy="none",
                use_decompositions=False,
                use_has_coloring_check=False,
                seed=rand.randint(0, 2**32 - 1),
            )
            counter = itertools.count()
            deque(zip(iterable, counter), maxlen=0)
            coloring_no = next(counter) // 2
            stats[coloring_no] += 1

    benchmark.pedantic(perform_test, rounds=rounds, warmup_rounds=0)

    data = np.array(list(stats.items()))
    df = DataFrame(data, columns=["coloring_cnt", "graph_cnt"])
    df.sort_values(by="coloring_cnt", inplace=True)
    df["graph_cnt"] //= rounds
    print(
        f"Most colorings: {tuple(df.iloc[-1])}, Most common: {tuple(df.loc[df["graph_cnt"].idxmax()])} (coloring_cnt, graph_cnt)"
    )
    # print(df.tail(n=50))

    df.to_csv("./benchmarks/results/laman_degree_3_plus_{}.csv".format(vertex_no))


################################################################################
# Fuzzy tests
################################################################################
@pytest.mark.slow
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
# @pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES)
@pytest.mark.parametrize("relabel_strategy", ["random"])
@pytest.mark.parametrize("graph", small_graphs[:SMALL_GRAPH_FUZZY_LIMIT])
def test_NAC_coloring_fuzzy_small_graphs(
    algorithm: str, graph: Graph, relabel_strategy: str
):
    """
    Checks algorithm validity against the naive implementation
    (that is hopefully correct) and checks that outputs are the same.
    Large number of smaller graphs is used (naive implementation is really slow for larger ones).
    """

    rand = random.Random(42)

    # print(graph)
    naive = list(
        graph.NAC_colorings(
            algorithm="naive",
            remove_vertices_cnt=0,
            # without this naive search is really slow
            # use_chromatic_partitions=False,
            use_decompositions=False,
            relabel_strategy="none",
            use_has_coloring_check=False,
        )
    )
    tested = list(
        graph.NAC_colorings(
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_has_coloring_check=False,
            seed=rand.randint(0, 1024),
        )
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


################################################################################
def test_wtf_is_going_on():
    import time
    from tqdm import tqdm

    rand = random.Random(42)

    limit = 1_000_000
    # limit = 10_000
    # limit = 1_000

    # dataset: List[Graph] = list(
    #     filter(lambda g: nx.number_of_nodes(g) == 15, laman_medium_graphs)
    # )[:30][
    #     15:16
    # ]  # [7:8]
    # dataset = general_graphs[:10]
    # dataset = general_graphs[6:7]
    # dataset = [general_graphs[i] for i in [6, 10, 13]]
    dataset: List[Graph] = list(
        filter(lambda g: nx.number_of_nodes(g) == 15, laman_medium_graphs)
    )[:30]
    dataset = [
        dataset[8],
        # dataset[18],
        # dataset[20],
        # dataset[23],
        # dataset[24],
        # dataset[25],
        # dataset[28],
    ]

    for graph in tqdm(dataset):
        results = []
        seed = rand.randint(0, 2**32 - 1)

        def test(
            algorithm: str = "subgraphs",
            relabel_strategy: str = "none",
            use_decompositions: bool = True,
        ):
            print(f"Test {algorithm} {relabel_strategy}")
            start = time.time()
            data = []
            for i, c in zip(
                range(limit),
                graph.NAC_colorings(
                    algorithm=algorithm,
                    relabel_strategy=relabel_strategy,
                    use_decompositions=use_decompositions,
                    use_has_coloring_check=False,
                    seed=seed,
                ),
            ):
                data.append(c)
                # if i % 500 == 0:
                #     print(i)
            duration = time.time() - start
            results.append(duration)
            print(f"Duration: {duration} ({len(data)} colorings)")
            print_stats()
            print()

        test(
            algorithm="subgraphs-linear-none-4-smart",
            relabel_strategy="none",
        )

        print(results)
        # [print() for _ in range(5)]
