from typing import Hashable, Optional
from stablecut.types import StableCut
import networkx as nx


def stable_set_violation[T: Hashable](
    graph: nx.Graph[T], vertices: set[T]
) -> Optional[tuple[T, T]]:
    for v in vertices:
        for n in graph.neighbors(v):
            if n in vertices:
                return v, n
    return None


def is_stable_set[T: Hashable](graph: nx.Graph[T], vertices: set[T]) -> bool:
    return stable_set_violation(graph, vertices) is None


def is_cut_set[T: Hashable](
    graph: nx.Graph[T], vertices: set[T], copy: bool = True
) -> bool:
    if copy:
        graph = nx.Graph(graph)

    graph.remove_nodes_from(vertices)
    return not nx.is_connected(graph)


def is_cut_set_separating[T: Hashable](
    graph: nx.Graph[T], vertices: set[T], u: T, v: T, copy: bool = True
) -> bool:
    if copy:
        graph = nx.Graph(graph)

    graph.remove_nodes_from(vertices)
    components = nx.connected_components(graph)
    for c in components:
        if u in c and v in c:
            return False
    return True


def is_stable_cut_set[T: Hashable](
    graph: nx.Graph[T], vertices: set[T], copy: bool = True
) -> bool:
    return is_stable_set(graph, vertices) and is_cut_set(graph, vertices, copy=copy)


def is_stable_cut_set_separating[T: Hashable](
    graph: nx.Graph[T], vertices: set[T], u: T, v: T, copy: bool = True
) -> bool:
    return is_stable_set(graph, vertices) and is_cut_set_separating(
        graph, vertices, u, v, copy=copy
    )
