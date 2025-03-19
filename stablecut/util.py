from typing import Iterable, Optional, TypeVar, Union
import networkx as nx

from pyrigi.data_type import Vertex

from stablecut.types import VertexCut


def _to_vertices[T](vertices: Iterable[T] | VertexCut[T]) -> set[T]:
    if isinstance(vertices, set):
        return vertices
    if isinstance(vertices, VertexCut):
        return vertices.cut
    return set(vertices)


def stable_set_violation[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | VertexCut[T],
) -> Optional[tuple[T, T]]:
    vertices = _to_vertices(vertices)
    for v in vertices:
        for n in graph.neighbors(v):
            if n in vertices:
                return v, n
    return None


def is_stable_set[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | VertexCut[T],
) -> bool:
    return stable_set_violation(graph, vertices) is None


def is_cut_set[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | VertexCut[T],
    copy: bool = True,
) -> bool:
    vertices = _to_vertices(vertices)

    if copy:
        graph = nx.Graph(graph)

    graph.remove_nodes_from(vertices)
    return not nx.is_connected(graph)


def is_cut_set_separating[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | VertexCut[T],
    u: T,
    v: T,
    copy: bool = True,
) -> bool:
    """
    Checks if the given cut separates vertices u and v.
    If either of the vertices is contained in the set, exception is thrown
    """
    vertices = _to_vertices(vertices)

    if u in vertices:
        raise ValueError(f"u={u} is in the set")
    if v in vertices:
        raise ValueError(f"v={v} is in the set")

    if copy:
        graph = nx.Graph(graph)

    graph.remove_nodes_from(vertices)
    components = nx.connected_components(graph)
    for c in components:
        if u in c and v in c:
            return False
    return True


def is_stable_cut_set[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | VertexCut[T],
    copy: bool = True,
) -> bool:
    return is_stable_set(graph, vertices) and is_cut_set(graph, vertices, copy=copy)


def is_stable_cut_set_separating[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | VertexCut[T],
    u: T,
    v: T,
    copy: bool = True,
) -> bool:
    return is_stable_set(graph, vertices) and is_cut_set_separating(
        graph, vertices, u, v, copy=copy
    )
