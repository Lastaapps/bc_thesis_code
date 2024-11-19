from typing import *

import networkx as nx

from pyrigi.nac.data_type import NACColoring
from pyrigi.nac.search import NAC_colorings_impl
from pyrigi.nac.single import has_NAC_coloring_impl, single_NAC_coloring_impl
from pyrigi.nac.single import (
    has_cartesian_NAC_coloring_impl,
    single_cartesian_NAC_coloring_impl,
)


def NAC_colorings(
    graph: nx.Graph,
    algorithm: str = "subgraphs",
    relabel_strategy: str = "none",
    use_chromatic_partitions: bool = True,
    use_decompositions: bool = True,
    remove_vertices_cnt: int = 0,
    use_has_coloring_check: bool = True,
    seed: int | None = None,
) -> Iterable[NACColoring]:
    return NAC_colorings_impl(
        self=graph,
        algorithm=algorithm,
        relabel_strategy=relabel_strategy,
        use_chromatic_partitions=use_chromatic_partitions,
        use_decompositions=use_decompositions,
        is_cartesian=False,
        remove_vertices_cnt=remove_vertices_cnt,
        use_has_coloring_check=use_has_coloring_check,
        seed=seed,
    )


def cartesian_NAC_colorings(
    graph: nx.Graph,
    algorithm: str = "subgraphs",
    relabel_strategy: str = "none",
    use_chromatic_partitions: bool = True,
    use_decompositions: bool = True,
    use_has_coloring_check: bool = True,
    seed: int | None = None,
) -> Iterable[NACColoring]:
    return NAC_colorings_impl(
        self=graph,
        algorithm=algorithm,
        relabel_strategy=relabel_strategy,
        use_chromatic_partitions=use_chromatic_partitions,
        use_decompositions=use_decompositions,
        is_cartesian=True,
        remove_vertices_cnt=0,
        use_has_coloring_check=use_has_coloring_check,
        seed=seed,
    )


def has_NAC_coloring(graph: nx.Graph) -> bool:
    """
    Same as single_NAC_coloring, but the certificate may not be created,
    so some additional tricks are used the performance may be improved.
    """
    return has_NAC_coloring_impl(graph)


def single_NAC_coloring(
    graph: nx.Graph,
    algorithm: str = "subgraphs",
) -> Optional[NACColoring]:
    return single_NAC_coloring_impl(graph, algorithm)


def has_cartesian_NAC_coloring(self) -> bool:
    """
    Same as single_castesian_NAC_coloring,
    but the certificate may not be created,
    so some additional tricks are used the performance may be improved.
    """
    return has_cartesian_NAC_coloring_impl(self)


def single_cartesian_NAC_coloring(
    graph: nx.Graph,
    algorithm: str | None = "subgraphs",
) -> Optional[NACColoring]:
    """
    Finds only a single NAC coloring if it exists.
    Some other optimizations may be used
    to improve performance for some graph classes.
    """
    return single_cartesian_NAC_coloring_impl(graph, algorithm)
