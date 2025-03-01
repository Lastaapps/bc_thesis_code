import logging
from typing import Hashable, Optional

from more_itertools import partition

from stablecut.types import StableCut
from stablecut.util import stable_set_violation
import networkx as nx
import numpy as np


def stable_cut_in_flexible_graph[T: Hashable](
    graph: nx.Graph[T],
    u: Optional[T] = None,
    v: Optional[T] = None,
    copy: bool = True,
) -> Optional[StableCut[T]]:
    """
    Finds a stable cut in a flexible graph
    according to Algorithm 1 in 2412.16018v1
    """

    # find the smallest connected component and put it first
    connected_components = list(nx.connected_components(graph))
    smallest_component_index = np.argmin(map(len, connected_components))
    connected_components[0], connected_components[smallest_component_index] = (
        connected_components[smallest_component_index],
        connected_components[0],
    )

    # if v is set, u must be also set
    if u is None and v is not None:
        u, v = v, u

    # choose a vertex at random
    if u is None:
        u = next(connected_components[0])
    else:
        assert u in graph

    if v is not None:
        assert v in graph

    # if the graph is not connected, we can possibly reduce the work needed
    # by finding a connected component that contains u
    # and finding a cut in it or just calling it the day
    # if v is not specified of lays in another component
    if len(connected_components) == 1:
        return stable_cut_in_flexible_graph_fast(graph, u, v, copy=copy)

    # find a connected component that contains u
    u_component, other = partition(lambda c: u in c, connected_components)
    u_component = next(u_component)
    subgraph = nx.induced_subgraph(graph, u_component)

    # if v is not specified, we just choose a different component and return the empty cut
    if v is None or v not in u_component:
        return StableCut(set(u_component), set(x for c in other for x in c), set())

    match _find_and_validate_u_and_v(subgraph, u, v):
        case None:
            return None
        case _:
            v = v

    if not copy:
        logging.warning("Copy is not avoidable for disconnected graphs")
    graph = nx.Graph(subgraph)

    cut = _process(graph, u, v)
    StableCut(a=cut.a, b=cut.b | set(x for c in other for x in c), cut=cut.cut)
    return cut


def stable_cut_in_flexible_graph_fast[T: Hashable](
    graph: nx.Graph[T],
    u: Optional[T] = None,
    v: Optional[T] = None,
    copy: bool = True,
    check_rigid_components: bool = True,
) -> Optional[StableCut[T]]:
    """
    Same as stable_cut_in_flexible_graph but faster.
    Checks for connectivity are removed, the algorithm may fail in those cases

    Parameters
    ----------
        copy - whether to make a copy of the graph before destructive modifications
        check_rigid_components - whether to check if u and v are in the same component
            to disable this check both u and v must be set
    """
    # if v is set, u must be also set
    if u is None and v is not None:
        u, v = v, u

    # choose a vertex at random
    if u is None:
        u = next(graph.nodes())

    # check is disabled => v must be set
    assert check_rigid_components or v is not None

    if check_rigid_components:
        match _find_and_validate_u_and_v(graph, u, v):
            case None:
                return None
            case _:
                v = v

    # graph will be modified in place
    if copy:
        graph = nx.Graph(graph)

    return _process(graph, u, v)


def _find_and_validate_u_and_v[T: Hashable](
    graph: nx.Graph[T],
    u: T,
    v: Optional[T],
) -> Optional[T]:
    from pyrigi import Graph as prGraph

    rigid_components = prGraph.rigid_components(graph)

    if len(rigid_components) < 2:
        logging.warning("Provided graph is not flexible")
        return None

    # vertices that share a component with u
    # for flexible graphs there should be more
    disallowed = set(v for c in rigid_components for v in c if u in c)

    # Check that input is valid
    if v is not None:
        if v in disallowed:
            logging.warning(f"Both vertices {u} and {v} share a same rigid component")
            return None
    else:
        # choose a vertex at random
        v = next(x for x in graph.nodes if x not in disallowed)
    return v


def _process[T: Hashable](
    graph: nx.Graph[T],
    u: T,
    v: T,
) -> StableCut[T]:
    from pyrigi import Graph as prGraph

    neiborhood = set(graph.neighbors(u))
    violation = stable_set_violation(graph, neiborhood)

    # found a stable set around u
    if violation is None:
        return StableCut(
            neiborhood | {u},
            set(graph.nodes) - {u},
            neiborhood,
        )

    def contract(graph: nx.Graph[T], u: T, x: T) -> tuple[set[T], set[T]]:
        """
        Contracts the vertices u and x
        and returns their original neighbors for easy restoration
        """
        u_neigh = set(graph.neighbors(u))
        x_neigh = set(graph.neighbors(x))
        graph.remove_node(x)
        for n in x_neigh:
            graph.add_edge(u, n)
        return u_neigh, x_neigh

    def restore(graph: nx.Graph[T], u: T, x: T, u_neigh: set[T], x_neigh: set[T]):
        """
        Restores contracted graph to it's original form
        """
        for n in x_neigh - u_neigh:
            graph.remove_edge(u, n)
        graph.add_node(x)
        for n in x_neigh:
            graph.add_edge(x, n)

    for x in violation:
        u_neigh, x_neigh = contract(graph, u, x)

        rigid_components = prGraph.rigid_components(graph)

        # The contracted vertex is in the same rigid component as v
        for c2 in filter(lambda c: v in c, rigid_components):
            if u in c2:
                restore(graph, u, x, u_neigh, x_neigh)
                continue

        return _process(graph, u, v)

    raise RuntimeError("Rigid components are not maximal")
