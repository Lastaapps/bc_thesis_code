################################################################################
# Monochromatic classes
################################################################################

from collections import defaultdict
import random
from typing import Dict, Iterable, List
from benchmarks.dataset import (
    load_laman_degree_3_plus,
    load_laman_graphs,
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


def group_by_monochomatic_classes_no[T: nx.Graph](
    graphs: Iterable[T],
) -> Dict[int, List[T]]:
    res: defaultdict[int, List[T]] = defaultdict(list)
    for graph in graphs:
        classes = len(Graph._find_triangle_components(graph)[1])
        res[classes].append(graph)
    return res


laman_graphs: List[Graph] = list(load_laman_graphs())
laman_degree_3_plus_graphs: List[Graph] = list(load_laman_degree_3_plus())

laman_grouped_classes = group_by_monochomatic_classes_no(laman_graphs)
laman_degree_3_plus_grouped_classes = group_by_monochomatic_classes_no(
    laman_degree_3_plus_graphs
)
print("Laman mon. classes groups:", sorted([(y, len(x)) for y, x in laman_grouped_classes.items()]))
print("Laman with deg >= 3 mon. classes groups:", sorted([(y, len(x)) for y, x in laman_degree_3_plus_grouped_classes.items()]))


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
            # "log_reverse",
            "sorted_bits",
            "sorted_size",
            "score",
            # "recursion",
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
        # higher numbers make no sense as we would fall back to the naive algorithm
        # for the monochromatic classes counts bellow
        for size in [3, 4, 5]
    ] + [
        # ("none", "subgraphs-linear-none-6"),
        # ("none", "subgraphs-linear-none-8"),
    ]
    RELABLE_STRATEGIES = [
        "none",
        "random",
        "bfs",
        # "beam_degree",
    ]

    BENCH_ROUNDS = 1


    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(60 * BENCH_ROUNDS)
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
        dataset = laman_grouped_classes[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough Laman graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
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
            # "cycles",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_cycle",
            "neighbors_degree",
            # "neighbors_degree_cycle",
            # "neighbors_iterative",
        ]
        for size in [3, 4, 5, 6, 7]
    ] + [
        # ("none", "subgraphs-linear-none-6"),
        # ("none", "subgraphs-linear-none-8"),
    ]

    BENCH_ROUNDS = 1


    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(15 * BENCH_ROUNDS)
    @pytest.mark.parametrize(("relabel_strategy", "algorithm"), ALGORITHMS)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt", "first_n"),
        [
            (14, 24, 1024),
            (15, 24, 1024),
            (16, 24, 1024),
            (17, 24, 1024),
            (18, 24, 1024),
            (19, 24, 1024),
            (20, 24, 1024),
            (21, 24, 1024),
            (22, 24, 1024),
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
        dataset = laman_grouped_classes[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough Laman graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
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
            "log_reverse",
            "sorted_bits",
            "sorted_size",
            "score",
            # "recursion",
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
        # higher numbers make no sense as we would fall back to the naive algorithm
        # for the monochromatic classes counts bellow
        for size in [3, 4, 5]
    ] + [
        # ("none", "subgraphs-linear-none-6"),
        # ("none", "subgraphs-linear-none-8"),
    ]
    RELABLE_STRATEGIES = [
        "none",
        "random",
        "bfs",
        # "beam_degree",
    ]

    BENCH_ROUNDS = 3


    @pytest.mark.nac_benchmark
    @pytest.mark.timeout(60 * BENCH_ROUNDS)
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    @pytest.mark.parametrize("relabel_strategy", RELABLE_STRATEGIES)
    @pytest.mark.parametrize(
        ("components_no", "graph_cnt"),
        [
            (7,  12),
            (8,  12),
            (9,  12),
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
        dataset = laman_degree_3_plus_grouped_classes[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough Laman graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
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
            # "cycles",
            "cycles_match_chunks",
            "neighbors",
            "neighbors_cycle",
            "neighbors_degree",
            # "neighbors_degree_cycle",
            # "neighbors_iterative",
        ]
        for size in [3, 4, 5, 6, 7]
    ] + [
        # ("none", "subgraphs-linear-none-6"),
        # ("none", "subgraphs-linear-none-8"),
    ]

    BENCH_ROUNDS = 3


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
            (21, 10, 1024),
            (22, 10, 1024),
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
        dataset = laman_degree_3_plus_grouped_classes[components_no]
        if len(dataset) < graph_cnt:
            raise ValueError(
                f"Not enough Laman graphs with {components_no} classes, required {graph_cnt} graphs, got {len(dataset)}"
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
