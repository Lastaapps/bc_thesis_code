################################################################################
# Monochromatic classes
################################################################################

from collections import defaultdict
import random
from typing import Dict, Iterable, List, TypedDict
from benchmarks.dataset import (
    GraphFamily,
    generate_dense_graphs,
    generate_sparse_graphs,
    load_all_small_graphs,
    load_general_graphs,
    load_generated_graphs,
    load_laman_degree_3_plus,
    load_laman_graphs,
    load_medium_generated_graphs,
    load_no_3_nor_4_cycle_graphs,
    load_small_generated_graphs,
    load_small_graph,
)
from pyrigi.graph import Graph
import pytest
import networkx as nx


################################################################################
# Data loading & Preprocessing
################################################################################
def group_by_vertex_no[T: nx.Graph](graphs: Iterable[T]) -> Dict[int, List[T]]:
    res: defaultdict[int, List[T]] = defaultdict(list)
    for graph in graphs:
        res[nx.number_of_nodes(graph)].append(graph)
    return res


def group_by_monochomatic_classes_no[
    T: nx.Graph
](graphs: Iterable[T],) -> Dict[int, List[T]]:
    res: defaultdict[int, List[T]] = defaultdict(list)
    for graph in graphs:
        classes = len(Graph._find_triangle_components(graph)[1])
        res[classes].append(graph)
    return res


class GraphLists:
    laman: List[Graph] = list(load_laman_graphs())
    laman_degree_3_plus: List[Graph] = list(load_laman_degree_3_plus())
    redundantly_rigid: List[Graph] = load_all_small_graphs(1024)

    general_small = load_small_generated_graphs().items()
    general_medium = (
        load_general_graphs("simple") | load_medium_generated_graphs()
    ).items()

    general_graphs = load_general_graphs("hard") | load_generated_graphs()
    general_quick = ["complete_multipartite", "self_complementary", "highly_irregular"]
    general_disabled = ["kneser"]  # + general_quick
    for key in general_disabled:
        general_graphs.pop(key, None)
    general_graphs = general_graphs.items()

    no_3_nor_4_cycles = load_no_3_nor_4_cycle_graphs()
    random_sparse = generate_sparse_graphs(20, 40)
    # this makes no sense as dense graphs form only a single class most of the time
    # random_dense = generate_dense_graphs(25, 40)


class GraphClasses:
    laman = group_by_monochomatic_classes_no(
        GraphLists.laman,
    )
    laman_degree_3_plus = group_by_monochomatic_classes_no(
        GraphLists.laman_degree_3_plus
    )
    redundantly_rigid = group_by_monochomatic_classes_no(
        GraphLists.redundantly_rigid,
    )
    no_3_nor_4_cycles = group_by_monochomatic_classes_no(
        GraphLists.no_3_nor_4_cycles,
    )
    random_sparse = group_by_monochomatic_classes_no(
        GraphLists.random_sparse,
    )

    @staticmethod
    def _sumarize(title: str, classes: Dict[int, List[Graph]]):
        print(
            title + ":\n",
            sorted([(y, len(x)) for y, x in classes.items()]),
            "\n",
        )

    @classmethod
    def print_sizes(cls):
        print()
        # GraphClasses._sumarize(
        #     "Laman mon. classes groups",
        #     cls.laman,
        # )
        # GraphClasses._sumarize(
        #     "Laman with deg >= 3 mon. classes groups",
        #     cls.laman_degree_3_plus,
        # )
        # GraphClasses._sumarize(
        #     "Redundantly rigid graphs",
        #     cls.redundantly_rigid,
        # )
        # GraphClasses._sumarize(
        #     "No 3 nor 4 cycles",
        #     cls.no_3_nor_4_cycles,
        # )
        # GraphClasses._sumarize(
        #     "Random sparse",
        #     cls.random_sparse,
        # )


GraphClasses.print_sizes()

# Laman mon. classes groups:
# [(2, 316), (3, 157), (4, 204), (5, 97), (6, 152), (7, 66), (8, 111), (9, 40),
# (10, 106), (11, 29), (12, 96), (13, 26), (14, 96), (15, 25), (16, 47), (17,
# 25), (18, 50), (19, 24), (20, 46), (21, 24), (22, 38), (23, 18), (24, 31),
# (25, 14), (26, 28), (27, 13), (28, 26), (29, 14), (30, 26), (31, 12), (32,
# 26), (33, 12), (34, 26), (35, 11), (36, 26), (37, 11), (38, 26), (39, 11),
# (40, 26), (41, 11), (42, 25), (43, 11), (44, 24), (45, 11), (46, 21), (47,
# 11), (48, 16), (49, 11), (50, 12), (51, 11), (52, 7), (53, 11), (55, 12),
# (57, 12)]
#
# Laman with deg >= 3 mon. classes groups:
# [(3, 4), (4, 5), (5, 19), (6, 14), (7, 19), (8, 12), (9, 46), (10, 23), (11,
# 43), (12, 12), (13, 56), (14, 16), (15, 59), (17, 75), (19, 77), (20, 11),
# (21, 103), (23, 135), (25, 77), (27, 128), (29, 128), (31, 128)]
#
# Redundantly rigid graphs:
# [(1, 9620), (2, 2439), (3, 471), (4, 978), (5, 445), (6, 647), (7, 289), (8,
# 404), (9, 317), (10, 223), (11, 154), (12, 185), (13, 70), (14, 47), (15,
# 55), (16, 19), (17, 13), (18, 24), (19, 14), (20, 10), (21, 8), (22, 4), (23,
# 2), (24, 2)]
#
# No 3 nor 4 cycles:
# [(10, 1), (12, 1), (15, 1), (16, 3), (18, 7), (21, 1), (23, 4), (26, 1), (28,
# 22), (31, 14), (57, 6), (87, 12)]
#
# Random sparse(25, 40):
# [(17, 3), (18, 4), (19, 3), (20, 7), (21, 11), (22, 9), (23, 21), (24, 17),
# (25, 14), (26, 22), (27, 23), (28, 20), (29, 29), (30, 29), (31, 32), (32,
# 38), (33, 44), (34, 34), (35, 36), (36, 26), (37, 29), (38, 27), (39, 28),
# (40, 24), (41, 28), (42, 26), (43, 26), (44, 28), (45, 38), (46, 27), (47,
# 24), (48, 25), (49, 35), (50, 33), (51, 30), (52, 18), (53, 12), (54, 15),
# (55, 19), (56, 18), (57, 19), (58, 12), (59, 15), (60, 8), (61, 10), (62, 6),
# (63, 4), (64, 3), (65, 3), (66, 4), (67, 1), (69, 1), (70, 2), (71, 2), (73,
# 1), (80, 1)]
#
# Random sparse(20, 40):
# [(9, 1), (10, 1), (11, 4), (12, 9), (13, 13), (14, 10), (15, 14), (16, 9),
# (17, 31), (18, 28), (19, 37), (20, 35), (21, 32), (22, 40), (23, 39), (24,
# 30), (25, 34), (26, 37), (27, 29), (28, 33), (29, 31), (30, 32), (31, 44),
# (32, 37), (33, 29), (34, 29), (35, 33), (36, 26), (37, 38), (38, 32), (39,
# 29), (40, 32), (41, 30), (42, 37), (43, 32), (44, 32), (45, 24), (46, 29),
# (47, 29), (48, 18), (49, 26), (50, 20), (51, 21), (52, 22), (53, 24), (54,
# 15), (55, 17), (56, 16), (57, 15), (58, 14), (59, 13), (60, 9), (61, 9), (62,
# 8), (63, 10), (64, 4), (65, 3), (66, 1), (67, 1), (68, 3), (70, 2), (72, 2)]


################################################################################
# Common logic
################################################################################
def _common_classes_all(
    benchmark,
    benchmark_rounds: int,
    dataset: List[Graph],
    algorithm: str,
    relabel_strategy: str,
    use_decompositions: bool = True,
):
    _common_classes_first(
        benchmark=benchmark,
        benchmark_rounds=benchmark_rounds,
        dataset=dataset,
        # I don't expect to have such an efficient algorithm it can handle this
        first_n=2**42,
        algorithm=algorithm,
        relabel_strategy=relabel_strategy,
        use_decompositions=use_decompositions,
    )


def _common_classes_first(
    benchmark,
    benchmark_rounds: int,
    dataset: List[Graph],
    first_n: int,
    algorithm: str,
    relabel_strategy: str,
    use_decompositions: bool = True,
):
    rand = random.Random(42)
    print()

    # dataset = [Graph(g) for g in dataset]

    from tqdm import tqdm

    def perform_test():

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

    benchmark.pedantic(perform_test, rounds=benchmark_rounds, warmup_rounds=0)


################################################################################
# Implementations
################################################################################


################################################################################
# Laman graphs
################################################################################
class TestLamanAll:
    # pairs (algorithm, relabel)
    ALGORITHMS = [
        "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
        for merge in [
            "linear",
            "log",
            "sorted_bits",
            "sorted_size",
            "score",
        ]
        for algo in [
            "none",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_cycle",
            "neighbors_degree",
        ]
        # higher numbers make no sense as we would fall back to the naive algorithm
        # for the monochromatic classes counts bellow
        for size in [3, 4, 5]
    ] + [
        # ("none", "subgraphs-linear-none-4"),
    ]
    RELABLE_STRATEGIES = [
        "none",
        "random",
        "bfs",
        # "beam_degree",
    ]

    BENCH_ROUNDS = 1

    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(15 * BENCH_ROUNDS)
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    @pytest.mark.parametrize("relabel_strategy", RELABLE_STRATEGIES)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt"),
        [
            (7, 24),
            (8, 24),
            (9, 24),
            (10, 24),
            (11, 24),
            (12, 24),
            (13, 24),
            (14, 24),
        ],
    )
    def test_LamanAll(
        self,
        benchmark,
        algorithm: str,
        relabel_strategy: str,
        components_no: int,
        graph_cnt: int,
        use_decompositions: bool = True,
    ):
        dataset = GraphClasses.laman[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
            )
        dataset = dataset[:graph_cnt]

        _common_classes_all(
            benchmark=benchmark,
            benchmark_rounds=self.BENCH_ROUNDS,
            dataset=dataset,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
        )


class TestLamanLarge:
    # pairs (algorithm, relabel)
    ALGORITHMS = [
        "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
        for merge in [
            "linear",
            "log",
            "sorted_bits",
            "sorted_size",
            "score",
        ]
        for algo in [
            "none",
            "random",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_degree",
        ]
        # higher numbers make no sense as we would fall back to the naive algorithm
        # for the monochromatic classes counts bellow
        for size in [3, 4, 5, 6]
    ] + [
        # ("none", "subgraphs-linear-none-4"),
    ]
    RELABLE_STRATEGIES = [
        "none",
        "random",
        "bfs",
        # "beam_degree",
    ]

    BENCH_ROUNDS = 1

    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(120 * BENCH_ROUNDS)
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    @pytest.mark.parametrize("relabel_strategy", RELABLE_STRATEGIES)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt"),
        [
            (15, 24),  # ~1.4 it/s
            (16, 12),  # ~1.8 it/s
            (17, 8),  # ~4.9 s/it
            # (18, 4),
        ],
    )
    def test_LamanLarge(
        self,
        benchmark,
        algorithm: str,
        relabel_strategy: str,
        components_no: int,
        graph_cnt: int,
        use_decompositions: bool = True,
    ):
        dataset = GraphClasses.laman[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
            )
        dataset = dataset[:graph_cnt]

        _common_classes_all(
            benchmark=benchmark,
            benchmark_rounds=self.BENCH_ROUNDS,
            dataset=dataset,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
        )


class TestLamanFirstN:
    # pairs (algorithm, relabel)
    ALGORITHMS = [
        ("random", "subgraphs-{}-{}-{}-smart".format(merge, algo, size))
        for merge in [
            "linear",
            "log",
            "sorted_bits",
        ]
        for algo in [
            "none",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_degree",
        ]
        for size in [4, 5, 6]
    ] + [
        ("none", "subgraphs-linear-none-4"),
    ]

    BENCH_ROUNDS = 1

    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(15 * BENCH_ROUNDS)
    @pytest.mark.parametrize(("relabel_strategy", "algorithm"), ALGORITHMS)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt", "first_n"),
        [
            (15, 24, 1024),
            (16, 24, 1024),
            (17, 24, 1024),
            (18, 24, 1024),
            (19, 24, 1024),
            (20, 24, 1024),
            (21, 14, 512),
            (22, 14, 512),
            (23, 14, 512),
            (24, 14, 512),
            (25, 14, 512),
            (26, 14, 512),
            (27, 14, 512),
            (28, 14, 512),
            (29, 14, 512),
            (30, 8, 256),
            (31, 8, 256),
            (32, 8, 256),
            (33, 8, 256),
            (34, 8, 256),
            (35, 8, 256),
            (36, 8, 256),
            (37, 8, 256),
            (38, 8, 256),
            (39, 8, 256),  # ~ 5-7 it/s
            (40, 6, 256),
            (41, 6, 256),
            (42, 6, 256),
            (43, 6, 256),
            (44, 6, 256),
            (45, 6, 256),
            (46, 6, 256),
            (47, 6, 256),
            (48, 6, 256),
            (49, 6, 256),
            (50, 6, 128),
            (51, 6, 128),
            (52, 6, 128),
            (53, 6, 128),
            (55, 6, 128),
            (57, 6, 128),
        ],
    )
    def test_LamanFirstN(
        self,
        benchmark,
        algorithm: str,
        relabel_strategy: str,
        components_no: int,
        graph_cnt: int,
        first_n: int,
        use_decompositions: bool = True,
    ):
        dataset = GraphClasses.laman[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
            )
        dataset = dataset[:graph_cnt]

        _common_classes_first(
            benchmark=benchmark,
            benchmark_rounds=self.BENCH_ROUNDS,
            dataset=dataset,
            first_n=first_n,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
        )


################################################################################
# Laman graphs with degree at least 3
################################################################################
class TestLamanDeg3PlusAll:
    # pairs (algorithm, relabel)
    ALGORITHMS = [
        "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
        for merge in [
            "linear",
            "log",
            "sorted_bits",
            "sorted_size",
            "score",
        ]
        for algo in [
            "none",
            "random",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_cycle",
            "neighbors_degree",
        ]
        # higher numbers make no sense as we would fall back to the naive algorithm
        # for the monochromatic classes counts bellow
        for size in [3, 4, 5]
    ] + [
        # ("none", "subgraphs-linear-none-4"),
    ]
    RELABLE_STRATEGIES = [
        "none",
        "random",
        "bfs",
        # "beam_degree",
    ]

    BENCH_ROUNDS = 3

    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(15 * BENCH_ROUNDS)
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    @pytest.mark.parametrize("relabel_strategy", RELABLE_STRATEGIES)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt"),
        [
            (7, 12),
            (8, 12),
            (9, 12),
            (10, 12),
            (11, 12),
            (12, 12),
            (13, 12),
        ],
    )
    def test_LamanDeg3PlusAll(
        self,
        benchmark,
        algorithm: str,
        relabel_strategy: str,
        components_no: int,
        graph_cnt: int,
        use_decompositions: bool = True,
    ):
        dataset = GraphClasses.laman_degree_3_plus[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
            )
        dataset = dataset[:graph_cnt]

        _common_classes_all(
            benchmark=benchmark,
            benchmark_rounds=self.BENCH_ROUNDS,
            dataset=dataset,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
        )


class TestLamanDeg3PlusFirstN:
    # pairs (algorithm, relabel)
    ALGORITHMS = [
        ("random", "subgraphs-{}-{}-{}-smart".format(merge, algo, size))
        for merge in [
            "linear",
            "log",
            "sorted_bits",
        ]
        for algo in [
            "none",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_cycle",
            "neighbors_degree",
        ]
        for size in [4, 5, 6]
    ] + [
        ("none", "subgraphs-linear-none-4"),
    ]

    BENCH_ROUNDS = 2

    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(15 * BENCH_ROUNDS)
    @pytest.mark.parametrize(("relabel_strategy", "algorithm"), ALGORITHMS)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt", "first_n"),
        [
            (14, 10, 1024),
            (15, 10, 1024),
            (16, 10, 1024),
            (17, 10, 1024),
            (18, 10, 1024),
            (19, 10, 1024),
            (20, 10, 1024),
            (21, 10, 1024),  # ~4-7 it/s
            (23, 10, 512),  # ~6-9 it/s
            (25, 10, 512),  # ~5-7 it/s
            (27, 10, 512),  # ~3-5 it/s
            (29, 10, 512),  # ~2-5 it/s
            (31, 10, 512),  # ~2-9 it/s
        ],
    )
    def test_LamanDeg3PlusFirstN(
        self,
        benchmark,
        algorithm: str,
        relabel_strategy: str,
        components_no: int,
        graph_cnt: int,
        first_n: int,
        use_decompositions: bool = True,
    ):
        dataset = GraphClasses.laman_degree_3_plus[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
            )
        dataset = dataset[:graph_cnt]

        _common_classes_first(
            benchmark=benchmark,
            benchmark_rounds=self.BENCH_ROUNDS,
            dataset=dataset,
            first_n=first_n,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
        )


################################################################################
# Redundantly Rigid graphs
################################################################################
class TestRedundantlyRigidAllSmall:
    # pairs (algorithm, relabel)
    ALGORITHMS = [
        "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
        for merge in [
            "linear",
            "log",
            "sorted_bits",
            "sorted_size",
            "score",
        ]
        for algo in [
            "none",
            "random",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_cycle",
            "neighbors_degree",
        ]
        # higher numbers make no sense as we would fall back to the naive algorithm
        # for the monochromatic classes counts bellow
        for size in [3, 4, 5]
    ] + [
        # ("none", "subgraphs-linear-none-4"),
    ]
    RELABLE_STRATEGIES = [
        "none",
        "random",
        "bfs",
        # "beam_degree",
    ]

    BENCH_ROUNDS = 1

    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(15 * BENCH_ROUNDS)
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    @pytest.mark.parametrize("relabel_strategy", RELABLE_STRATEGIES)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt"),
        [
            (7, 100),
            (8, 100),
            (9, 100),
            (10, 100),
            (11, 100),
            (12, 100),
            (13, 64),
            (14, 42),
        ],
    )
    def test_RedundantlyRigidAll(
        self,
        benchmark,
        algorithm: str,
        relabel_strategy: str,
        components_no: int,
        graph_cnt: int,
        use_decompositions: bool = True,
    ):
        dataset = GraphClasses.redundantly_rigid[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
            )
        dataset = dataset[:graph_cnt]

        _common_classes_all(
            benchmark=benchmark,
            benchmark_rounds=self.BENCH_ROUNDS,
            dataset=dataset,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
        )


class TestRedundantlyRigidAllLarge:
    # pairs (algorithm, relabel)
    ALGORITHMS = [
        "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
        for merge in [
            "linear",
            "log",
            "sorted_bits",
            "sorted_size",
            "score",
        ]
        for algo in [
            "none",
            "random",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_cycle",
            "neighbors_degree",
        ]
        # higher numbers make no sense as we would fall back to the naive algorithm
        # for the monochromatic classes counts bellow
        for size in [3, 4, 5]
    ] + [
        # ("none", "subgraphs-linear-none-4"),
    ]
    RELABLE_STRATEGIES = [
        "none",
        "random",
        "bfs",
        # "beam_degree",
    ]

    BENCH_ROUNDS = 1

    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(15 * BENCH_ROUNDS)
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    @pytest.mark.parametrize("relabel_strategy", RELABLE_STRATEGIES)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt"),
        [
            (15, 48),
            (16, 16),
            (17, 12),
            (18, 24),
            (19, 14),
            (20, 10),
            (21, 8),
            (22, 4),
            (23, 2),
            (24, 2),
        ],
    )
    def test_RedundantlyRigidAll(
        self,
        benchmark,
        algorithm: str,
        relabel_strategy: str,
        components_no: int,
        graph_cnt: int,
        use_decompositions: bool = True,
    ):
        dataset = GraphClasses.redundantly_rigid[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
            )
        dataset = dataset[:graph_cnt]

        _common_classes_all(
            benchmark=benchmark,
            benchmark_rounds=self.BENCH_ROUNDS,
            dataset=dataset,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
        )


################################################################################
# No 3 nor 4 cycles
################################################################################
class TestNo3Nor4CyclesFirstN:
    # pairs (algorithm, relabel)
    ALGORITHMS = [
        ("random", "subgraphs-{}-{}-{}-smart".format(merge, algo, size))
        for merge in [
            "linear",
            "log",
            "sorted_bits",
        ]
        for algo in [
            "none",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_cycle",
            "neighbors_degree",
        ]
        for size in [4, 5, 6]
    ] + [
        # ("none", "subgraphs-linear-none-4"),
    ]

    BENCH_ROUNDS = 3

    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(30 * BENCH_ROUNDS)
    @pytest.mark.parametrize(("relabel_strategy", "algorithm"), ALGORITHMS)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt", "first_n"),
        [
            (16, 3, 1024),
            (18, 7, 1024),
            (28, 22, 512),  # 3 it/s
            (31, 14, 512),
            (57, 6, 128),
            (87, 12, 64),
        ],
    )
    def test_LamanNo3nor4CyclesFirstN(
        self,
        benchmark,
        algorithm: str,
        relabel_strategy: str,
        components_no: int,
        graph_cnt: int,
        first_n: int,
        use_decompositions: bool = True,
    ):
        dataset = GraphClasses.no_3_nor_4_cycles[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
            )
        dataset = dataset[:graph_cnt]

        _common_classes_first(
            benchmark=benchmark,
            benchmark_rounds=self.BENCH_ROUNDS,
            dataset=dataset,
            first_n=first_n,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
        )


################################################################################
# Sparse graphs
################################################################################
class TestSparseAll:
    # pairs (algorithm, relabel)
    ALGORITHMS = [
        "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
        for merge in [
            "linear",
            "log",
            "sorted_bits",
            "sorted_size",
            "score",
        ]
        for algo in [
            "none",
            "random",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_cycle",
            "neighbors_degree",
        ]
        # higher numbers make no sense as we would fall back to the naive algorithm
        # for the monochromatic classes counts bellow
        for size in [3, 4, 5]
    ] + [
        # ("none", "subgraphs-linear-none-4"),
    ]
    RELABLE_STRATEGIES = [
        "none",
        "random",
        "bfs",
        # "beam_degree",
    ]

    BENCH_ROUNDS = 1

    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(15 * BENCH_ROUNDS)
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    @pytest.mark.parametrize("relabel_strategy", RELABLE_STRATEGIES)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt"),
        [
            (13, 10),
            (14, 10),
            (15, 10),
            (16, 9),
            (17, 20),
        ],
    )
    def test_SparseAll(
        self,
        benchmark,
        algorithm: str,
        relabel_strategy: str,
        components_no: int,
        graph_cnt: int,
        use_decompositions: bool = True,
    ):
        dataset = GraphClasses.random_sparse[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
            )
        dataset = dataset[:graph_cnt]

        _common_classes_all(
            benchmark=benchmark,
            benchmark_rounds=self.BENCH_ROUNDS,
            dataset=dataset,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
        )


class TestSparseFirstN:
    # pairs (algorithm, relabel)
    ALGORITHMS = [
        ("random", "subgraphs-{}-{}-{}-smart".format(merge, algo, size))
        for merge in [
            "linear",
            "log",
            "sorted_bits",
        ]
        for algo in [
            "cycles_match_chunks",
            "neighbors",
            "neighbors_degree",
        ]
        for size in [4, 5, 6]
    ] + [
        ("none", "subgraphs-linear-none-4"),
    ]

    BENCH_ROUNDS = 2

    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(15 * BENCH_ROUNDS)
    @pytest.mark.parametrize(("relabel_strategy", "algorithm"), ALGORITHMS)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt", "first_n"),
        [
            (18, 20, 1024),
            (19, 20, 1024),
            (20, 20, 1024),
            (21, 20, 1024),
            (22, 20, 1024),
            (23, 20, 1024),
            (24, 20, 1024),
            (25, 20, 1024),
            (26, 20, 1024),
            (27, 20, 1024),
            (28, 20, 1024),
            (29, 20, 1024),
            (30, 20, 512),
            (31, 20, 512),
            (32, 20, 512),
            (33, 20, 512),
            (34, 20, 512),
            (35, 20, 512),
            (36, 20, 512),
            (37, 20, 512),
            (38, 20, 512),
            (39, 20, 512),
            (40, 20, 512),
            (41, 20, 512),
            (42, 20, 512),
            (43, 20, 512),
            (44, 20, 512),
            (45, 20, 512),
            (46, 20, 512),
            (47, 20, 512),
            (48, 20, 512),
            (49, 20, 512),
            (50, 10, 256),
            (51, 10, 256),
            (52, 10, 256),
            (53, 10, 256),
            (54, 10, 256),
            (55, 10, 256),
            (56, 10, 256),
            (57, 10, 256),
            (58, 10, 256),
            (59, 10, 256),
        ],
    )
    def test_SparseFirstN(
        self,
        benchmark,
        algorithm: str,
        relabel_strategy: str,
        components_no: int,
        graph_cnt: int,
        first_n: int,
        use_decompositions: bool = True,
    ):
        dataset = GraphClasses.random_sparse[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
            )
        dataset = dataset[:graph_cnt]

        _common_classes_first(
            benchmark=benchmark,
            benchmark_rounds=self.BENCH_ROUNDS,
            dataset=dataset,
            first_n=first_n,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
        )
