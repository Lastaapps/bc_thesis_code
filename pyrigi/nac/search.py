# TODO NAC clear imports
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from functools import reduce
from itertools import combinations
import itertools
import random
from typing import (
    Callable,
    Collection,
    Deque,
    Iterable,
    List,
    Any,
    Literal,
    Sequence,
    Union,
    Tuple,
    Optional,
    Dict,
    Set,
)

import networkx as nx
from sympy import Matrix
import math
import numpy as np
import time

from pyrigi.datastructures.union_find import UnionFind
from pyrigi.misc import doc_category, generate_category_tables
from pyrigi.exception import LoopError
from pyrigi.util.lazy_product import lazy_product
from pyrigi.util.repetable_iterator import RepeatableIterator


from pyrigi.nac.data_type import NACColoring, Edge
from pyrigi.nac.monochromatic_classes import (
    find_triangle_components,
    fake_triangle_components,
    create_T_graph_from_components,
)
from pyrigi.nac.existence import has_NAC_coloring_checks, check_NAC_constrains
from pyrigi.nac.check import _is_cartesian_NAC_coloring_impl, _is_NAC_coloring_impl
from pyrigi.nac.development import (
    NAC_PRINT_SWITCH,
    NAC_statistics_generator,
    NAC_statistics_colorings_merge_wrapper,
)
from pyrigi.nac.util import NiceGraph


def _coloring_from_mask(
    vertices: List[int],
    mask: int,
    component_to_edges: List[List[Edge]],
    allow_mask: int | None = None,
) -> NACColoring:

    # TODO use numpy and boolean addressing
    if allow_mask is None:
        allow_mask = 2 ** len(vertices) - 1

    red, blue = [], []  # set(), set()
    for i, e in enumerate(vertices):
        address = 1 << i

        if address & allow_mask == 0:
            continue

        edges = component_to_edges[e]
        # (red if mask & address else blue).update(edges)
        (red if mask & address else blue).extend(edges)
    return (red, blue)

    # numpy impl, ~10% slower
    # if allow_mask is not None:
    #     vertices = vertices[allow_mask]
    #     mask = mask[allow_mask]

    # red_vert = vertices[mask]
    # blue_vert = vertices[~mask]

    # red = [edge for edges in red_vert for edge in component_to_edges[edges]]
    # blue = [edge for edges in blue_vert for edge in component_to_edges[edges]]

    # return (red, blue)


def _NAC_colorings_naive(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    check_selected_NAC_coloring: Callable[[nx.Graph, NACColoring], bool],
) -> Iterable[NACColoring]:
    vertices = list(t_graph.nodes())

    # iterate all the coloring variants
    # division by 2 is used as the problem is symmetrical
    for mask in range(1, 2 ** len(vertices) // 2):
        coloring = _coloring_from_mask(vertices, mask, component_to_edges)

        if not check_selected_NAC_coloring(graph, coloring):
            continue

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])


def find_cycles_in_T_graph(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    from_angle_preserving_components: bool,
    all: bool = False,
) -> Set[Tuple[int, ...]]:
    """
    For each vertex finds one/all of the shortest cycles it lays on

    Parameters
    ----------
    graph:
        The graph to work with, vertices should be integers indexed from 0.
        Usually a graphs with triangle and angle preserving components.
    all:
        if set to True, all the shortest cycles are returned
        Notice that for dense graphs the number of cycles can be quite huge
        if set to False, some cycle is returned for each vertex
        Defaults to False
    ----------
    """
    found_cycles: Set[Tuple[int, ...]] = set()

    vertices = list(t_graph.nodes)

    # disables vertices that represent a disconnected angle preserving
    # component. Search using these components is then disabled.
    # This prevents some smallest cycles to be found.
    # On the other hand cartesian NAC coloring is not purpose of this
    # work, so I don't care about peek performance yet.
    disabled_vertices: Set[int] = set()

    if from_angle_preserving_components:
        for v in vertices:
            g = graph.edge_subgraph(component_to_edges[v])
            if not nx.connected.is_connected(g):
                disabled_vertices.add(v)

    def insert_found_cycle(cycle: List[int]) -> None:
        """
        Runs post-processing on a cycle
        Makes sure the first element is the smallest one
        and the second one is greater than the last one.
        This is required for equality checks in set later.
        """

        # find the smallest element index
        smallest = 0
        for i, e in enumerate(cycle):
            if e < cycle[smallest]:
                smallest = i

        # makes sure that the element following the smallest one
        # is greater than the one preceding it
        if cycle[smallest - 1] < cycle[(smallest + 1) % len(cycle)]:
            cycle = list(reversed(cycle))
            smallest = len(cycle) - smallest - 1

        # rotates the list so the smallest element is first
        cycle = cycle[smallest:] + cycle[:smallest]

        found_cycles.add(tuple(cycle))

    def bfs(start: int) -> None:
        """
        Finds the shortest cycle(s) for the vertex given
        """
        # Stores node and id of branch it's from
        queue: deque[Tuple[int, int]] = deque()
        # this wastes a little, but whatever
        parent_and_id = [(-1, -1) for _ in range(max(vertices) + 1)]
        parent_and_id[start] = (start, -1)
        local_cycle_len = -1

        for u in t_graph.neighbors(start):
            if u in disabled_vertices:
                continue

            # newly found item
            parent_and_id[u] = (start, -u)
            queue.append((u, -u))

        def backtrack(v: int, u: int) -> List[int]:
            """
            Reconstructs the found cycle
            """
            cycles: List[int] = []

            # reconstructs one part of the cycle
            cycles.append(u)
            p = parent_and_id[u][0]
            while p != start:
                cycles.append(p)
                p = parent_and_id[p][0]
            cycles = list(reversed(cycles))

            # and the other one
            cycles.append(v)
            p = parent_and_id[v][0]
            while p != start:
                cycles.append(p)
                p = parent_and_id[p][0]

            cycles.append(start)
            return cycles

        # typical BFS
        while queue:
            v, id = queue.popleft()
            parent = parent_and_id[v][0]

            for u in t_graph.neighbors(v):
                # so I don't create cycle on 1 edge
                # this could be done sooner, I know...
                if u == parent:
                    continue

                if u in disabled_vertices:
                    continue

                # newly found item
                if parent_and_id[u][0] == -1:
                    parent_and_id[u] = (v, id)
                    queue.append((u, id))
                    continue

                # cycle like this one does not contain the starting vertex
                if parent_and_id[u][1] == id:
                    continue

                # a cycle was found
                cycle = backtrack(v, u)

                if local_cycle_len == -1:
                    local_cycle_len = len(cycle)

                # We are so far in the bfs process that all
                # the cycles will be longer now
                if len(cycle) > local_cycle_len:
                    return

                insert_found_cycle(cycle)

                if all:
                    continue
                else:
                    return

    for start in vertices:
        # bfs is a separate function so I can use return in it
        if start in disabled_vertices:
            continue
        bfs(start)

    return found_cycles


def _find_cycles(
    graph: nx.Graph,
    subgraph_components: Set[int],
    component_to_edges: List[Set[Tuple[int, int]]],
    all: bool = False,
) -> Set[Tuple[int, ...]]:
    match 2:
        case 0:
            return set()
        case 1:
            return find_shortest_cycles(
                graph, subgraph_components, component_to_edges, all
            )
        case 2:
            return find_useful_cycles(graph, subgraph_components, component_to_edges)
        case 3:
            res = find_shortest_cycles(
                graph, subgraph_components, component_to_edges, all
            ) | find_useful_cycles(graph, subgraph_components, component_to_edges)
            res = list(sorted(res, key=lambda x: len(x)))[: 2 * graph.number_of_nodes()]
            return set(res)


def find_shortest_cycles(
    graph: nx.Graph,
    subgraph_components: Set[int],
    component_to_edges: List[Set[Tuple[int, int]]],
    all: bool = False,
    per_class_limit: int = 1024,
) -> Set[Tuple[int, ...]]:
    cycles = find_shortest_cycles_for_components(
        graph=graph,
        subgraph_components=subgraph_components,
        component_to_edges=component_to_edges,
        all=all,
        per_class_limit=per_class_limit,
    )
    return {c for comp_cycles in cycles.values() for c in comp_cycles}


def find_shortest_cycles_for_components(
    graph: nx.Graph,
    subgraph_components: Set[int],
    component_to_edges: List[Set[Tuple[int, int]]],
    all: bool = False,
    per_class_limit: int = 1024,
) -> Dict[int, Set[Tuple[int, ...]]]:
    """
    For each edge finds one/all of the shortest cycles it lays on

    Parameters
    ----------
    graph:
        The graph to work with, vertices should be integers indexed from 0.
        Usually a graphs with triangle and angle preserving components.
    all:
        if set to True, all the shortest cycles are returned
        Notice that for dense graphs the number of cycles can be quite huge
        if set to False, some cycle is returned for each vertex
        Defaults to False
    ----------

    Return
    ------
        Component to the smallest cycle mapping, result is list of comp ids
    """
    found_cycles: Dict[int, Set[Tuple[int, ...]]] = defaultdict(set)

    edge_to_components: Dict[Tuple[int, int], int] = {
        e: comp_id for comp_id, comp in enumerate(component_to_edges) for e in comp
    }

    vertices = list(graph.nodes)

    def insert_found_cycle(comp_id: int, cycle: List[int]) -> None:
        """
        Runs post-processing on a cycle
        Makes sure the first element is the smallest one
        and the second one is greater than the last one.
        This is required for equality checks in set later.
        """

        # find the smallest element index
        smallest = 0
        for i, e in enumerate(cycle):
            if e < cycle[smallest]:
                smallest = i

        # makes sure that the element following the smallest one
        # is greater than the one preceding it
        if cycle[smallest - 1] < cycle[(smallest + 1) % len(cycle)]:
            cycle = list(reversed(cycle))
            smallest = len(cycle) - smallest - 1

        # rotates the list so the smallest element is first
        cycle = cycle[smallest:] + cycle[:smallest]

        found_cycles[comp_id].add(tuple(cycle))

    def bfs(start_component_id: int) -> None:
        """
        Finds the shortest cycle(s) for the edge given
        """
        start_component = component_to_edges[start_component_id]
        # print()
        # print()
        # print()
        # print()
        # print(f"{start_component=}")

        # Stores node and id of branch it's from
        queue: deque[Tuple[int, int]] = deque()
        # this wastes a little, but whatever (parent, branch_id)
        parent_and_id = [(-1, -1) for _ in range(max(vertices) + 1)]
        local_cycle_len = -1

        start_component_vertices: Set[int] = {v for e in start_component for v in e}

        def backtrack(v: int, u: int) -> List[int]:
            """
            Reconstructs the found cycle
            """
            cycle: List[int] = []

            # print(f"{parent_and_id=}")
            # print(f"comp_id={start_component_id} {v=} {u=}")
            # print(graphviz_components("crash", component_to_edges))
            # print(f"{component_to_edges=}")
            cycle.append(u)
            traversed = parent_and_id[u]
            while traversed[0] != -1:
                cycle.append(traversed[0])
                traversed = parent_and_id[traversed[0]]

            cycle = list(reversed(cycle))
            # print(f"{cycle=}")

            cycle.append(v)
            traversed = parent_and_id[v]
            while traversed[0] != -1:
                cycle.append(traversed[0])
                traversed = parent_and_id[traversed[0]]
            # print(f"{cycle=}")

            # cycle translated to comp ids
            comp_cycle: List[int] = [start_component_id]
            for i in range(len(cycle) - 1):
                edge: Tuple[int, int] = tuple(cycle[i : i + 2])
                comp_id: int = edge_to_components.get(
                    edge, edge_to_components.get((edge[1], edge[0]), None)
                )
                assert comp_id is not None
                if comp_cycle[-1] != comp_id:
                    comp_cycle.append(comp_id)

            # print(f"{comp_cycle=}")

            # assert len(comp_cycle) == len(set(comp_cycle))
            return comp_cycle

        def add_vertex(vfrom: int, vto: int) -> bool:
            """
            Return
            ------
                True if the search should continue
            """
            comp_id: int = edge_to_components.get(
                (vfrom, vto), edge_to_components.get((vto, vfrom), None)
            )
            if comp_id not in subgraph_components:
                return True

            nonlocal local_cycle_len
            branch_id = parent_and_id[vfrom][1]
            assert comp_id is not None
            edges = component_to_edges[comp_id]
            vertices = {v for e in edges for v in e}
            # print(f"Processing {vfrom=} {vto=}")

            q = deque([vfrom])
            while q:
                v = q.popleft()
                # print(f"Popping {v}")
                for u in graph.neighbors(v):
                    if u not in vertices:
                        continue

                    _, id = parent_and_id[u]
                    if id == -1:
                        q.append(u)
                        queue.append((u, branch_id))
                        parent_and_id[u] = (v, branch_id)
                        # print(f"Queing {v} -> {u}")
                        continue
                    elif id == branch_id:
                        continue

                    # print(f"Cycle! {v}->{u}")
                    # a cycle was found
                    cycle = backtrack(v, u)

                    if local_cycle_len == -1:
                        local_cycle_len = len(cycle)

                    # We are so far in the bfs process that all
                    # the cycles will be longer now
                    if len(cycle) > local_cycle_len:
                        return False

                    insert_found_cycle(start_component_id, cycle)

                    if all:
                        continue
                    else:
                        return False
            return True

        for v in start_component_vertices:
            parent_and_id[v] = (-1, v)
        for v in start_component_vertices:
            for u in graph.neighbors(v):
                if u in start_component_vertices:
                    continue
                if not add_vertex(v, u):
                    return

        # typical BFS
        while queue:
            v, id = queue.popleft()
            parent = parent_and_id[v][0]

            for u in graph.neighbors(v):
                if u == parent:
                    continue
                if not add_vertex(v, u):
                    return

    for component in range(len(component_to_edges)):
        # bfs is a separate function so I can use return in it
        if component not in subgraph_components:
            continue
        bfs(component)

    # return found_cycles

    limited = {}
    for key, value in found_cycles.items():
        limited[key] = set(list(sorted(value, key=lambda x: len(x)))[:per_class_limit])
    return limited


def find_useful_cycles(
    graph: nx.Graph,
    subgraph_components: Set[int],
    component_to_edges: List[Set[Tuple[int, int]]],
    per_class_limit: int = 2,
) -> Set[Tuple[int, ...]]:
    cycles = find_useful_cycles_for_components(
        graph=graph,
        subgraph_components=subgraph_components,
        component_to_edges=component_to_edges,
        per_class_limit=per_class_limit,
    )
    return {c for comp_cycles in cycles.values() for c in comp_cycles}


def find_useful_cycles_for_components(
    graph: nx.Graph,
    subgraph_components: Set[int],
    component_to_edges: List[Set[Tuple[int, int]]],
    per_class_limit: int = 2,
) -> Dict[int, Set[Tuple[int, ...]]]:
    """
    Not all the returned cycles are guaranteed to be actual cycles as
    this may create cycles that enter and exit a component at the same vertex.
    """
    comp_no = len(component_to_edges)
    vertex_to_components = [set() for _ in range(max(graph.nodes) + 1)]
    for comp_id, comp in enumerate(component_to_edges):
        if comp_id not in subgraph_components:
            continue
        for u, v in comp:
            vertex_to_components[u].add(comp_id)
            vertex_to_components[v].add(comp_id)
    neighboring_components = [set() for _ in range(comp_no)]

    found_cycles: Dict[int, Set[Tuple[int, ...]]] = defaultdict(set)

    for v in graph.nodes:
        for i in vertex_to_components[v]:
            for j in vertex_to_components[v]:
                if i != j:
                    neighboring_components[i].add(j)

    def insert_cycle(comp_id: int, cycle: Tuple[int, ...]):
        # in case one component was used more times
        if len(set(cycle)) != len(cycle):
            return

        # find the smallest element index
        smallest = 0
        for i, e in enumerate(cycle):
            if e < cycle[smallest]:
                smallest = i

        # makes sure that the element following the smallest one
        # is greater than the one preceding it
        if cycle[smallest - 1] < cycle[(smallest + 1) % len(cycle)]:
            cycle = list(reversed(cycle))
            smallest = len(cycle) - smallest - 1

        # rotates the list so the smallest element is first
        cycle = cycle[smallest:] + cycle[:smallest]

        # print(f"Inserting [{comp_id}] -> {cycle}")
        found_cycles[comp_id].add(tuple(cycle))

    # print()
    # print(f"{subgraph_components=}")
    # print(f"{component_to_edges=}")
    # print(f"{vertex_to_components=}")
    # print(f"{neighboring_components=}")

    for u, v in graph.edges:
        # print(f"{u=} {v=}")
        u_comps = vertex_to_components[u]
        v_comps = vertex_to_components[v]

        # remove shared components
        intersection = u_comps.intersection(v_comps)
        u_comps = u_comps - intersection
        v_comps = v_comps - intersection
        # print(f"{u_comps=} {v_comps=} <-> {intersection=}")
        assert len(intersection) <= 1
        if len(intersection) == 0:
            continue

        for u_comp in u_comps:
            # triangles
            for n in neighboring_components[u_comp].intersection(v_comps):
                for i in intersection:
                    insert_cycle(i, (i, u_comp, n))

            for v_comp in v_comps:
                # squares
                u_comp_neigh = neighboring_components[u_comp]
                v_comp_neigh = neighboring_components[v_comp]
                # print(f"{u_comp_neigh=} {v_comp_neigh=}")
                res = u_comp_neigh.intersection(v_comp_neigh) - intersection
                for i in intersection:
                    for r in res:
                        insert_cycle(i, (i, u_comp, r, v_comp))

                # pentagons
                for r in u_comp_neigh - set([u_comp]):
                    for t in neighboring_components[r].intersection(v_comp_neigh):
                        for i in intersection:
                            insert_cycle(i, (i, u_comp, r, t, v_comp))

    limited = {}
    for key, value in found_cycles.items():
        limited[key] = set(list(sorted(value, key=lambda x: len(x)))[:per_class_limit])

    return limited


def _create_bitmask_for_t_graph_cycle(
    graph: nx.Graph,
    component_to_edges: Callable[[int], List[Edge]],
    cycle: Tuple[int, ...],
    local_vertices: Set[int] | None = None,
) -> Tuple[int, int]:
    """
    Creates a bit mask (template) to match vertices in the cycle
    and a mask matching vertices of the cycle that make NAC coloring
    impossible if they are the only ones with different color

    Parameters
    ----------
    component_to_edges:
        Mapping from component to it's edges. Can be list.__getitem__.
    local_vertices:
        can be used if the graph given is subgraph of the original graph
        and component_to_edges also represent the original graph.
    ----------
    """

    template = 0
    valid = 0
    # template = np.zeros(len(vertices), dtype=np.bool)
    # valid = np.zeros(len(vertices), dtype=np.bool)

    for v in cycle:
        template |= 1 << v
        # template[v] = True

    def check_for_connecting_edge(prev: int, curr: int, next: int) -> bool:
        """
        Checks if for the component given (curr) exists path trough the
        component using single edge only - in that case color change of
        the triangle component can ruin NAC coloring.
        """
        # print(f"{prev=} {curr=} {next=}")
        vertices_curr = {v for e in component_to_edges(curr) for v in e}

        # You may think that if the component is a single edge,
        # it must connect the circle. Because we are using a t-graph,
        # which is based on the line graph idea, the edge can share
        # a vertex with both the neighboring components.
        # An example for this is a star with 3 edges.

        vertices_prev = {v for e in component_to_edges(prev) for v in e}
        vertices_next = {v for e in component_to_edges(next) for v in e}
        # print(f"{vertices_prev=} {vertices_curr=} {vertices_next=}")
        intersections_prev = vertices_prev.intersection(vertices_curr)
        intersections_next = vertices_next.intersection(vertices_curr)
        # print(f"{intersections_prev=} {intersections_next=}")

        if local_vertices is not None:
            intersections_prev = intersections_prev.intersection(local_vertices)
            intersections_next = intersections_next.intersection(local_vertices)

        for p in intersections_prev:
            neighbors = set(graph.neighbors(p))
            for n in intersections_next:
                if n in neighbors:
                    return True
        return False

    for prev, curr, next in zip(cycle[-1:] + cycle[:-1], cycle, cycle[1:] + cycle[:1]):
        if check_for_connecting_edge(prev, curr, next):
            valid |= 1 << curr
            # valid[curr] = True

    # print(cycle, bin(template), bin(valid))
    return template, valid


def _mask_matches_templates(
    templates: List[Tuple[int, int]],
    mask: int,
    subgraph_mask: int,
) -> bool:
    """
    Checks if mask given matches any of the cycles given.

    Parameters
    ----------
        templates:
            list of outputs of the _create_bitmask_for_t_graph_cycle
            graph method - mask representing vertices presence in a cycle
            and validity mask noting which of them exclude NAC coloring
            if present.
        mask:
            bit mask of vertices in the same order that
            are currently red (or blue).
        subgraph_mask:
            bit mask representing all the vertices of the current subgraph.
    ----------
    """
    for template, validity in templates:
        stamp1, stamp2 = mask & template, (mask ^ subgraph_mask) & template
        cnt1, cnt2 = stamp1.bit_count(), stamp2.bit_count()
        stamp, cnt = (stamp1, cnt1) if cnt1 == 1 else (stamp2, cnt2)

        if cnt != 1:
            continue

        # now we know there is one node that has a wrong color
        # we check if the node is a triangle component
        # if so, we need to know it if also ruins the coloring
        if stamp & validity:
            return True
    return False


def _NAC_colorings_cycles(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    check_selected_NAC_coloring: Callable[[nx.Graph, NACColoring], bool],
    from_angle_preserving_components: bool,
    use_all_cycles: bool = False,
) -> Iterable[NACColoring]:
    vertices = list(t_graph.nodes())
    # so we start with 0
    vertices.sort()

    # find some small cycles for state filtering
    cycles = _find_cycles(
        graph,
        set(t_graph.nodes),
        component_to_edges,
        all=use_all_cycles,
    )
    # the idea is that smaller cycles reduce the state space more
    cycles = sorted(cycles, key=lambda c: len(c))

    templates = [
        _create_bitmask_for_t_graph_cycle(graph, component_to_edges.__getitem__, c)
        for c in cycles
    ]
    templates = [t for t in templates if t[1] > 0]

    # if len(cycles) != 0:
    #     templates_and_validities = [create_bitmask(c) for c in cycles]
    #     templates, validities = zip(*templates_and_validities)
    #     templates = np.stack(templates)
    #     validities = np.stack(validities)
    # else:
    #     templates = np.empty((0, len(vertices)), dtype=np.bool)
    #     validities = np.empty((0, len(vertices)), dtype=np.bool)

    # this is used for mask inversion, because how ~ works on python
    # numbers, if we used some kind of bit arrays,
    # this would not be needed.
    subgraph_mask = 0  # 2 ** len(vertices) - 1
    for v in vertices:
        subgraph_mask |= 1 << v
    # subgraph_mask = np.ones(len(vertices), dtype=np.bool)
    # demasking = 2**np.arange(len(vertices))

    # iterate all the coloring variants
    # division by 2 is used as the problem is symmetrical
    for mask in range(1, 2 ** len(vertices) // 2):

        # This is part of a slower implementation using numpy
        # TODO remove before the final merge
        # my_mask = (demasking & mask).astype(np.bool)
        # def check_cycles(my_mask: np.ndarray) -> bool:
        #     # we mask the cycles
        #     masked = templates & my_mask
        #     usable_rows = np.sum(masked, axis=-1) == 1
        #     valid = masked & validities
        #     return bool(np.any(valid[usable_rows]))
        # if check_cycles(my_mask) or check_cycles(my_mask ^ subgraph_mask):
        #     continue

        if _mask_matches_templates(templates, mask, subgraph_mask):
            continue

        coloring = _coloring_from_mask(vertices, mask, component_to_edges)

        if not check_selected_NAC_coloring(graph, coloring):
            continue

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])


# ensures components that are later joined are next to each other
# increasing the chance of NAC colorings being dismissed sooner


def _split_and_find(
    local_graph: nx.Graph,
    local_chunk_sizes: List[int],
    search_func: Callable[[nx.Graph, Sequence[int]], List[int]],
) -> List[int]:
    if len(local_chunk_sizes) <= 2:
        return search_func(local_graph, local_chunk_sizes)

    length = len(local_chunk_sizes)
    sizes = (
        sum(local_chunk_sizes[: length // 2]),
        sum(local_chunk_sizes[length // 2 :]),
    )
    ordered_vertices = search_func(local_graph, sizes)
    groups = (ordered_vertices[: sizes[0]], ordered_vertices[sizes[0] :])
    graphs = tuple(nx.induced_subgraph(local_graph, g) for g in groups)
    assert len(graphs) == 2
    return _split_and_find(
        graphs[0], local_chunk_sizes[: length // 2], search_func
    ) + _split_and_find(graphs[1], local_chunk_sizes[length // 2 :], search_func)


def _subgraphs_strategy_degree_cycles(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
) -> List[int]:
    degree_ordered_vertices = _degree_ordered_nodes(t_graph)
    vertex_cycles = _cycles_per_vertex(graph, t_graph, component_to_edges)

    ordered_vertices: List[int] = []
    used_vertices: Set[int] = set()
    for v in degree_ordered_vertices:
        # Handle vertices with no cycles
        if v not in used_vertices:
            ordered_vertices.append(v)
            used_vertices.add(v)

        for cycle in vertex_cycles[v]:
            for u in cycle:
                if u in used_vertices:
                    continue
                ordered_vertices.append(u)
                used_vertices.add(u)
    return ordered_vertices


def _subgraphs_strategy_cycles(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
) -> List[int]:
    degree_ordered_vertices = _degree_ordered_nodes(t_graph)
    vertex_cycles = _cycles_per_vertex(graph, t_graph, component_to_edges)

    ordered_vertices: List[int] = []
    used_vertices: Set[int] = set()
    all_vertices = deque(degree_ordered_vertices)

    while all_vertices:
        v = all_vertices.popleft()

        # the vertex may have been used before from a cycle
        if v in used_vertices:
            continue

        queue: Deque[int] = deque([v])

        while queue:
            u = queue.popleft()

            if u in used_vertices:
                continue

            ordered_vertices.append(u)
            used_vertices.add(u)

            for cycle in vertex_cycles[u]:
                for u in cycle:
                    if u in used_vertices:
                        continue
                    queue.append(u)
    return ordered_vertices


def _subgraphs_strategy_cycles_match_chunks(
    chunk_sizes: Sequence[int],
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
) -> List[int]:
    chunk_size = chunk_sizes[0]
    degree_ordered_vertices = _degree_ordered_nodes(t_graph)
    vertex_cycles = _cycles_per_vertex(graph, t_graph, component_to_edges)

    ordered_vertices: List[int] = []
    used_vertices: Set[int] = set()
    all_vertices = deque(degree_ordered_vertices)

    while all_vertices:
        v = all_vertices.popleft()

        # the vertex may have been used before from a cycle
        if v in used_vertices:
            continue

        used_in_epoch = 0
        queue: Deque[int] = deque([v])

        while queue and used_in_epoch < chunk_size:
            u = queue.popleft()

            if u in used_vertices:
                continue

            ordered_vertices.append(u)
            used_vertices.add(u)
            used_in_epoch += 1

            for cycle in vertex_cycles[u]:
                for u in cycle:
                    if u in used_vertices:
                        continue
                    queue.append(u)

        while used_in_epoch < chunk_size and len(used_vertices) != len(
            degree_ordered_vertices
        ):
            v = all_vertices.pop()
            if v in used_vertices:
                continue
            ordered_vertices.append(v)
            used_vertices.add(v)
            used_in_epoch += 1

    for v in degree_ordered_vertices:
        if v in used_vertices:
            continue
        ordered_vertices.append(v)

    return ordered_vertices


def _subgraphs_strategy_bfs(
    t_graph: nx.Graph,
    chunk_sizes: Sequence[int],
) -> List[int]:
    graph = nx.Graph(t_graph)
    used_vertices: Set[int] = set()
    ordered_vertices_groups: List[List[int]] = [[] for _ in chunk_sizes]

    for v in _degree_ordered_nodes(graph):
        if v in used_vertices:
            continue

        index_min = min(
            range(len(ordered_vertices_groups)),
            key=lambda x: len(ordered_vertices_groups[x]) / chunk_sizes[x],
        )
        target = ordered_vertices_groups[index_min]

        added_vertices: List[int] = [v]
        used_vertices.add(v)
        target.append(v)

        for _, u in nx.bfs_edges(graph, v):
            if u in used_vertices:
                continue

            added_vertices.append(u)
            target.append(u)
            used_vertices.add(u)

            if len(target) == chunk_sizes[index_min]:
                break

        graph.remove_nodes_from(added_vertices)

    return [v for group in ordered_vertices_groups for v in group]


def _subgraphs_strategy_neighbors(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    chunk_sizes: Sequence[int],
    iterative: bool,
    start_with_cycle: bool,
    use_degree: bool,
    seed: int | None,
) -> List[int]:
    t_graph = nx.Graph(t_graph)
    ordered_vertices_groups: List[List[int]] = [[] for _ in chunk_sizes]

    # if False, chunk does need to assign random component
    is_random_component_required: List[bool] = [True for _ in chunk_sizes]

    edge_to_components: Dict[Tuple[int, int], int] = {
        e: comp_id for comp_id, comp in enumerate(component_to_edges) for e in comp
    }
    # print(f"{component_to_edges=}")

    # start algo and fill a chunk
    # profit
    while t_graph.number_of_nodes() > 0:
        # TODO connected components
        # TODO run algorithm per component, not per vertex
        # TODO omit adding vertices with only single connection when
        #      the chunk is almost full

        component_ids = list(t_graph.nodes)
        rand = random.Random(seed)
        rand_comp = component_ids[rand.randint(0, len(component_ids) - 1)]

        # could by avoided by having a proper subgraph
        local_vertices: Set[int] = {
            v
            for comp_id, comp in enumerate(component_to_edges)
            for e in comp
            for v in e
            if comp_id in component_ids
        }
        # print(f"{component_ids=}")
        # print(f"{local_vertices=}")

        chunk_index = min(
            range(len(ordered_vertices_groups)),
            key=lambda x: len(ordered_vertices_groups[x]) / chunk_sizes[x],
        )
        target = ordered_vertices_groups[chunk_index]
        # print(f"Open target {chunk_index}, {len(target)}/{chunk_sizes[chunk_index]}")

        if is_random_component_required[chunk_index]:
            if start_with_cycle:
                cycles = _find_cycles(
                    graph,
                    set(t_graph.nodes),
                    component_to_edges,
                    all=True,
                )
                shortest: List[Tuple[int, ...]] = []
                shortest_len = 2**42
                for cycle in cycles:
                    if len(cycle) == shortest_len:
                        shortest.append(cycle)
                    elif len(cycle) < shortest_len:
                        shortest = [cycle]
                        shortest_len = len(cycle)

                # no cycles found of the cycles are to long
                if len(cycles) == 0 or shortest_len > chunk_sizes[chunk_index] - len(
                    target
                ):
                    added_components: Set[int] = set([rand_comp])
                    target.append(rand_comp)
                else:
                    cycle = shortest[rand.randint(0, len(shortest) - 1)]
                    added_components = set([c for c in cycle])
                    target.extend(cycle)
            else:
                # print(f"Rand comp: {rand_comp} -> {component_to_edges[rand_comp]}")
                added_components: Set[int] = set([rand_comp])
                target.append(rand_comp)
            is_random_component_required[chunk_index] = False
        else:
            added_components = set()

        used_vertices = {
            v for comp in target for e in component_to_edges[comp] for v in e
        }
        opened: Set[int] = set()

        for v in used_vertices:
            for u in graph.neighbors(v):
                if u in used_vertices:
                    continue
                if u not in local_vertices:
                    continue
                opened.add(u)

        iteration_no = 0
        while (
            opened
            and len(target) < chunk_sizes[chunk_index]
            and (iteration_no <= 2 or not iterative)
        ):
            comp_added = False

            if not use_degree:
                values = [
                    (u, len(used_vertices.intersection(graph.neighbors(u))))
                    for u in opened
                ]
            else:
                values = [
                    (
                        u,
                        (
                            len(used_vertices.intersection(graph.neighbors(u))),
                            # degree
                            -len(local_vertices.intersection(graph.neighbors(u))),
                        ),
                    )
                    for u in opened
                ]

            # TODO handle maybe
            # rand.shuffle(values)

            # print(f"{values=}")
            best_vertex = max(values, key=lambda x: x[1])[0]
            # print(f"Choosen {best_vertex}")
            for neighbor in used_vertices.intersection(graph.neighbors(best_vertex)):
                if neighbor not in local_vertices:
                    # print("X: Non-local")
                    continue

                comp_id: int = edge_to_components.get(
                    (best_vertex, neighbor),
                    edge_to_components.get((neighbor, best_vertex), None),
                )
                if comp_id not in component_ids:
                    # print("X: Invalid component")
                    continue
                if comp_id in added_components:
                    # print("X: Already added")
                    continue

                # print(f"Adding component ({best_vertex} -- {neighbor}) {comp_id} -> {component_to_edges[comp_id]}")
                added_components.add(comp_id)
                target.append(comp_id)
                comp_added = True

                if len(target) >= chunk_sizes[chunk_index]:
                    # print("Chunk is full")
                    break

                new_vertices: Set[int] = {
                    v for e in component_to_edges[comp_id] for v in e
                }
                used_vertices |= new_vertices
                opened -= new_vertices

                for v in new_vertices:
                    for u in graph.neighbors(v):
                        if u in used_vertices:
                            continue
                        if u not in local_vertices:
                            continue
                        opened.add(u)
            if comp_added:
                iteration_no += 1
            else:
                opened.remove(best_vertex)

        # Nothing happened, we need to find some component randomly
        if iteration_no == 0:
            is_random_component_required[chunk_index] = True

        t_graph.remove_nodes_from(added_components)
    return [v for group in ordered_vertices_groups for v in group]


def _wrong_subgraphs_strategy_beam_neighbors(
    t_graph: nx.Graph,
    chunk_sizes: Sequence[int],
    start_with_triangles: bool,
    start_from_min: bool,
) -> List[int]:
    t_graph = nx.Graph(t_graph)
    ordered_vertices_groups: List[List[int]] = [[] for _ in chunk_sizes]
    beam_size: int = min(chunk_sizes[0], 10)
    # beam_size: int = 1024

    while t_graph.number_of_nodes() > 0:
        if start_from_min:
            start = min(t_graph.degree(), key=lambda x: x[1])[0]
        else:
            start = max(t_graph.degree(), key=lambda x: x[1])[0]

        if start not in t_graph.nodes:
            continue

        queue: List[int] = [start]

        index_min = min(
            range(len(ordered_vertices_groups)),
            key=lambda x: len(ordered_vertices_groups[x]) / chunk_sizes[x],
        )
        target = ordered_vertices_groups[index_min]

        bfs_visited: Set[int] = set([start])
        added_vertices: Set[int] = set()

        # it's quite beneficial to start with a triangle
        # in fact we just apply the same strategy as later
        # just for the first vertex added as it has no context yet
        if start_with_triangles:
            start_neighbors = set(t_graph.neighbors(start))
            for neighbor in start_neighbors:
                if len(start_neighbors.intersection(t_graph.neighbors(neighbor))) > 0:
                    queue.append(neighbor)
                    bfs_visited.add(neighbor)
                    break

        while queue and len(target) < chunk_sizes[index_min]:
            # more neighbors are already part of the graph -> more better
            # also, this is asymptotically really slow,
            # but I'm not implementing smart heaps, this is python,
            # it's gonna be slow anyway (also the graphs are small)

            values = [
                len(added_vertices.intersection(t_graph.neighbors(u))) for u in queue
            ]
            # values = [(len(added_vertices.intersection(t_graph.neighbors(u))), -t_graph.degree(u)) for u in queue]

            sorted_by_metric = sorted(
                [i for i in range(len(values))],
                key=lambda i: values[i],
                reverse=True,
            )
            v = queue[sorted_by_metric[0]]
            queue = [queue[i] for i in sorted_by_metric[1 : beam_size + 1]]

            # this is worse, but somehow slightly more performant
            # but I'm not using it anyway ;)
            # largest = max(range(len(values)), key=values.__getitem__)
            # v = queue.pop(largest)
            # queue = queue[:beam_size]

            added_vertices.add(v)
            target.append(v)

            for u in t_graph.neighbors(v):
                if u in bfs_visited:
                    continue
                bfs_visited.add(u)
                queue.append(u)

        t_graph.remove_nodes_from(added_vertices)
    return [v for group in ordered_vertices_groups for v in group]


def _wrong_subgraphs_strategy_components(
    t_graph: nx.Graph,
    chunk_sizes: Sequence[int],
    start_from_biggest_component: bool,
) -> List[int]:
    chunk_no = len(chunk_sizes)
    # NetworkX crashes otherwise
    if t_graph.number_of_nodes() < 2:
        return list(t_graph.nodes())

    k_components = nx.connectivity.k_components(t_graph)

    if len(k_components) == 0:
        return list(t_graph.nodes())

    keys = sorted(k_components.keys(), reverse=True)

    if not start_from_biggest_component:
        for i, key in enumerate(keys):
            if len(k_components[key]) <= chunk_no:
                keys = keys[i:]
                break

    ordered_vertices: List[int] = []
    used_vertices: Set[int] = set()

    for key in keys:
        for component in k_components[key]:
            for v in component:
                if v in used_vertices:
                    continue
                ordered_vertices.append(v)
                used_vertices.add(v)

    # make sure all the nodes were added
    for v in t_graph.nodes():
        if v in used_vertices:
            continue
        ordered_vertices.append(v)

    return ordered_vertices


def _degree_ordered_nodes(graph: nx.Graph) -> List[int]:
    return list(
        map(
            lambda x: x[0],
            sorted(
                graph.degree(),
                key=lambda x: x[1],
                reverse=True,
            ),
        )
    )


def _cycles_per_vertex(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
) -> List[List[Tuple[int, ...]]]:
    """
    For each vertex we find all the cycles that contain it
    Finds all the shortest cycles in the graph
    """
    cycles = _find_cycles(
        graph,
        set(t_graph.nodes),
        component_to_edges,
        all=True,
    )

    # vertex_cycles = [[] for _ in range(t_graph.number_of_nodes())]
    vertex_cycles = [[] for _ in range(max(t_graph.nodes) + 1)]
    for cycle in cycles:
        for v in cycle:
            vertex_cycles[v].append(cycle)
    return vertex_cycles


def _subgraphs_join_epochs(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    check_selected_NAC_coloring: Callable[[nx.Graph, NACColoring], bool],
    from_angle_preserving_components: bool,
    vertices: List[int],
    epoch1: Iterable[int],
    subgraph_mask_1: int,
    epoch2: RepeatableIterator[int],
    subgraph_mask_2: int,
) -> Iterable[int]:
    """
    Joins almost NAC colorings of two disjoined subgraphs of a t-graph.
    This function works by doing a cross product of the almost NAC colorings
    on subgraphs, joining the subgraps into a single sub graph
    and checking if the joined colorings are still almost NAC.

    Almost NAC coloring is NAC coloring that is not necessarily surjective.

    Returns found almost NAC colorings and mask of the new subgraph
    """

    if subgraph_mask_1 & subgraph_mask_2:
        raise ValueError("Cannot join two subgraphs with common nodes")

    subgraph_mask = subgraph_mask_1 | subgraph_mask_2

    local_vertices: List[int] = []

    # local vertex -> global index
    mapping: Dict[int, int] = {}

    for i, v in enumerate(vertices):
        if (1 << i) & subgraph_mask:
            mapping[v] = i
            local_vertices.append(v)

    local_t_graph = nx.Graph(nx.induced_subgraph(t_graph, local_vertices))
    local_cycles = _find_cycles(
        graph,
        set(local_t_graph.nodes),
        component_to_edges,
    )

    mapped_components_to_edges = lambda ind: component_to_edges[vertices[ind]]
    # cycles with indices of the vertices in the global order
    local_cycles = [tuple(mapping[c] for c in cycle) for cycle in local_cycles]
    templates = [
        _create_bitmask_for_t_graph_cycle(graph, mapped_components_to_edges, cycle)
        for cycle in local_cycles
    ]
    templates = [t for t in templates if t[1] > 0]

    counter = 0
    if NAC_PRINT_SWITCH:
        print(
            f"Join started ({2**subgraph_mask_1.bit_count()}+{2**subgraph_mask_2.bit_count()}->{2**subgraph_mask.bit_count()})"
        )

    # in case lazy_product is removed, return nested fors as they are faster
    # and also return repeatable iterator requirement
    mask_iterator = ((mask1, mask2) for mask1 in epoch1 for mask2 in epoch2)

    # this prolongs the overall computation time,
    # but in case we need just a "small" number of colorings,
    # this can provide them faster
    # disabled as result highly depended on the graph given
    # TODO benchmark on larger dataset
    # mask_iterator = lazy_product(epoch1, epoch2)

    for mask1, mask2 in mask_iterator:
        mask = mask1 | mask2

        if _mask_matches_templates(templates, mask, subgraph_mask):
            continue

        coloring = _coloring_from_mask(
            vertices,
            mask,
            component_to_edges,
            subgraph_mask,
        )

        if not check_selected_NAC_coloring(graph, coloring):
            continue

        counter += 1
        yield mask

    if NAC_PRINT_SWITCH:
        print(f"Join yielded: {counter}")


@NAC_statistics_generator
def _subgraph_colorings_generator(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    check_selected_NAC_coloring: Callable[[nx.Graph, NACColoring], bool],
    vertices: List[int],
    chunk_size: int,
    offset: int,
) -> Iterable[int]:
    match 1:
        case 0:
            return _subgraph_colorings_cycles_generator(
                graph,
                t_graph,
                component_to_edges,
                check_selected_NAC_coloring,
                vertices,
                chunk_size,
                offset,
            )
        case 1:
            return _subgraph_colorings_removal_generator(
                graph,
                t_graph,
                component_to_edges,
                check_selected_NAC_coloring,
                vertices,
                chunk_size,
                offset,
            )


@NAC_statistics_generator
def _subgraph_colorings_cycles_generator(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    check_selected_NAC_coloring: Callable[[nx.Graph, NACColoring], bool],
    vertices: List[int],
    chunk_size: int,
    offset: int,
) -> Iterable[int]:
    """
    iterate all the coloring variants
    division by 2 is used as the problem is symmetrical
    """
    # The last chunk can be smaller
    local_vertices: List[int] = vertices[offset : offset + chunk_size]
    # print(f"Local vertices: {local_vertices}")

    local_t_graph = nx.Graph(nx.induced_subgraph(t_graph, local_vertices))
    local_cycles = _find_cycles(
        graph,
        set(local_t_graph.nodes),
        component_to_edges,
    )

    # local -> first chunk_size vertices
    mapping = {x: i for i, x in enumerate(local_vertices)}

    mapped_components_to_edges = lambda ind: component_to_edges[local_vertices[ind]]
    local_cycles = (tuple(mapping[c] for c in cycle) for cycle in local_cycles)
    templates = [
        _create_bitmask_for_t_graph_cycle(graph, mapped_components_to_edges, cycle)
        for cycle in local_cycles
    ]
    templates = [t for t in templates if t[1] > 0]

    counter = 0
    subgraph_mask = 2 ** len(local_vertices) - 1
    for mask in range(0, 2**chunk_size // 2):
        if _mask_matches_templates(templates, mask, subgraph_mask):
            continue

        coloring = _coloring_from_mask(local_vertices, mask, component_to_edges)

        if not check_selected_NAC_coloring(graph, coloring):
            continue

        counter += 1
        yield mask << offset

    if NAC_PRINT_SWITCH:
        print(f"Base yielded: {counter}")


@NAC_statistics_generator
def _subgraph_colorings_removal_generator(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    check_selected_NAC_coloring: Callable[[nx.Graph, NACColoring], bool],
    partitions: List[int],
    chunk_size: int,
    offset: int,
) -> Iterable[int]:
    """
    iterate all the coloring variants
    division by 2 is used as the problem is symmetrical
    """

    # TODO assert for cartesian NAC coloring

    local_partitions: List[int] = partitions[offset : offset + chunk_size]
    local_edges = [
        edge for comp_id in local_partitions for edge in component_to_edges[comp_id]
    ]
    graph = nx.edge_subgraph(graph, local_edges)

    def processor(graph: nx.Graph) -> Iterable[NACColoring]:
        # print(f"graph={Graph(graph)}")
        # return list(_NAC_colorings_without_vertex(processor, graph, None))
        return _NAC_colorings_without_vertex(processor, graph, None)

    nodes_iter = iter(local_partitions)
    # This makes sure no symmetric colorings are produced
    disabled_comp = next(nodes_iter)
    edge_map = defaultdict(int)
    disabled_edge = component_to_edges[disabled_comp][0]
    edge_map[(disabled_edge[0], disabled_edge[1])] = len(component_to_edges) + 1
    edge_map[(disabled_edge[1], disabled_edge[0])] = len(component_to_edges) + 1

    for comp_id in nodes_iter:
        edge = component_to_edges[comp_id][0]
        edge_map[(edge[0], edge[1])] = local_partitions.index(comp_id) + 1
        edge_map[(edge[1], edge[0])] = local_partitions.index(comp_id) + 1

    # print()
    # print(f"{component_to_edges=}")
    # print(f"{partitions=} {offset=} {chunk_size=}")
    # print(f"{edge_map=}")
    # produced = set()
    # print("Accepted", bin(0))
    # all_ones = 2**len(local_partitions) - 1

    counter = 0
    yield 0
    for red, blue in processor(graph):
        # TODO swith red and blue if blue is smaller
        # print(f"{red=} {blue=}")
        mask = 0
        invalid_coloring_found = False
        for edge in red:
            partition_ind = edge_map[edge]
            if partition_ind == 0:
                continue
            mask |= 1 << partition_ind
            partition_ind -= 1
            partition_ind = (partition_ind < len(local_partitions)) * partition_ind

            # make sure all the edges of each partition share the same component
            partition_id = local_partitions[partition_ind]
            for partiotion_edge in component_to_edges[partition_id]:
                if (
                    partiotion_edge not in red
                    and (partiotion_edge[1], partiotion_edge[0]) not in red
                ):
                    invalid_coloring_found = True
                    # print(f"Invalid red  coloring {partition_ind=}({edge_map[edge]-1}) -> {partition_id=}")
                    break
            if invalid_coloring_found:
                break

        for edge in blue:
            partition_ind = edge_map[edge]
            if partition_ind == 0:
                continue
            partition_ind -= 1
            partition_ind = (partition_ind < len(local_partitions)) * partition_ind

            # make sure all the edges of each partition share the same component
            partition_id = local_partitions[partition_ind]
            for partiotion_edge in component_to_edges[partition_id]:
                if (
                    partiotion_edge not in blue
                    and (partiotion_edge[1], partiotion_edge[0]) not in blue
                ):
                    invalid_coloring_found = True
                    # print(f"Invalid blue coloring {partition_ind=}({edge_map[edge]-1}) -> {partition_id=}")
                    break
            if invalid_coloring_found:
                break
        if invalid_coloring_found:
            continue

        mask >>= 1
        if mask & (1 << len(component_to_edges)):
            continue
        counter += 1
        mask <<= offset

        # print("Accepted", bin(mask))
        # coloring = _coloring_from_mask(
        #     local_partitions, mask >> offset, component_to_edges
        # )
        # print(f"{coloring=}")
        # if mask in produced:
        #     assert False
        # produced.add(mask)

        yield mask

    if NAC_PRINT_SWITCH:
        print(f"Base yielded: {counter}")


def _NAC_colorings_subgraphs(
    graph: nx.Graph,
    t_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    check_selected_NAC_coloring: Callable[[nx.Graph, NACColoring], bool],
    seed: int,
    from_angle_preserving_components: bool,
    get_subgraphs_together: bool = True,
    merge_strategy: (
        str | Literal["linear", "log", "log_reverse", "min_max", "weight"]
    ) = "log",
    preferred_chunk_size: int | None = None,
    order_strategy: str = "none",
) -> Iterable[NACColoring]:
    """
    This version of the algorithm splits the graphs into subgraphs,
    find NAC colorings for each of them. The subgraphs are then merged
    and new colorings are reevaluated till we reach the original graph again.
    The algorithm tries to find optimal subgraphs and merge strategy.
    """
    # These values are taken from benchmarks as (almost) optimal
    if preferred_chunk_size is None:
        # preferred_chunk_size = round(0.08 * t_graph.number_of_nodes() + 5.5)
        preferred_chunk_size = round(
            np.log(t_graph.number_of_nodes()) / np.log(1 + 1 / 2)
        )
        preferred_chunk_size = max(int(preferred_chunk_size), 4)

    preferred_chunk_size = min(preferred_chunk_size, t_graph.number_of_nodes())
    assert preferred_chunk_size >= 1

    # Represents size (no. of vertices of the t-graph) of a basic subgraph
    vertices_no = t_graph.number_of_nodes()

    def create_chunk_sizes() -> List[int]:
        """
        Makes sure all the chunks are the same size of 1 bigger

        Could be probably significantly simpler,
        like np.fill and add some bitmask, but this was my first idea,
        get over it.
        """
        # chunk_size = max(
        #     int(np.sqrt(vertices_no)), min(preferred_chunk_size, vertices_no)
        # )
        # chunk_no = (vertices_no + chunk_size - 1) // chunk_size
        chunk_no = vertices_no // preferred_chunk_size
        chunk_sizes = []
        remaining_len = vertices_no
        for _ in range(chunk_no):
            # ceiling floats, scary
            chunk_sizes.append(
                min(
                    math.ceil(remaining_len / (chunk_no - len(chunk_sizes))),
                    remaining_len,
                )
            )
            remaining_len -= chunk_sizes[-1]
        return chunk_sizes

    chunk_sizes = create_chunk_sizes()

    def process(
        search_func: Callable[[nx.Graph, Sequence[int]], List[int]],
    ):
        if get_subgraphs_together:
            return _split_and_find(
                t_graph,
                chunk_sizes,
                search_func,
            )
        else:
            return search_func(t_graph, chunk_sizes)

    match order_strategy:
        case "none":
            vertices = list(t_graph.nodes())

        case "random":
            vertices = list(t_graph.nodes())
            random.Random(seed).shuffle(vertices)

        case "degree":
            vertices = process(lambda g, _: _degree_ordered_nodes(g))

        case "degree_cycles":
            vertices = process(
                lambda g, _: _subgraphs_strategy_degree_cycles(
                    graph,
                    g,
                    component_to_edges,
                )
            )
        case "cycles":
            vertices = process(
                lambda g, _: _subgraphs_strategy_cycles(
                    graph,
                    g,
                    component_to_edges,
                )
            )
        case "cycles_match_chunks":
            vertices = process(
                lambda g, l: _subgraphs_strategy_cycles_match_chunks(
                    l,
                    graph,
                    g,
                    component_to_edges,
                )
            )
        case "bfs":
            vertices = process(
                lambda g, l: _subgraphs_strategy_bfs(
                    t_graph=g,
                    chunk_sizes=l,
                )
            )
        case "neighbors":
            vertices = process(
                lambda g, l: _subgraphs_strategy_neighbors(
                    graph=graph,
                    t_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=False,
                    iterative=False,
                    use_degree=False,
                    seed=seed,
                )
            )
        case "neighbors_cycle" | "neighbors_cycles":
            vertices = process(
                lambda g, l: _subgraphs_strategy_neighbors(
                    graph=graph,
                    t_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=True,
                    iterative=False,
                    use_degree=False,
                    seed=seed,
                )
            )
        case "neighbors_degree":
            vertices = process(
                lambda g, l: _subgraphs_strategy_neighbors(
                    graph=graph,
                    t_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=False,
                    iterative=False,
                    use_degree=True,
                    seed=seed,
                )
            )
        case "neighbors_degree_cycle":
            vertices = process(
                lambda g, l: _subgraphs_strategy_neighbors(
                    graph=graph,
                    t_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=True,
                    iterative=False,
                    use_degree=True,
                    seed=seed,
                )
            )
        case "neighbors_iterative":
            vertices = process(
                lambda g, l: _subgraphs_strategy_neighbors(
                    graph=graph,
                    t_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=False,
                    iterative=True,
                    use_degree=False,
                    seed=seed,
                )
            )
        case "neighbors_iterative_cycle":
            vertices = process(
                lambda g, l: _subgraphs_strategy_neighbors(
                    graph=graph,
                    t_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=True,
                    iterative=True,
                    use_degree=False,
                    seed=seed,
                )
            )
        case "beam_neighbors":
            vertices = process(
                lambda g, l: _wrong_subgraphs_strategy_beam_neighbors(
                    t_graph=g,
                    chunk_sizes=l,
                    start_with_triangles=False,
                    start_from_min=True,
                )
            )
        case "beam_neighbors_max":
            vertices = process(
                lambda g, l: _wrong_subgraphs_strategy_beam_neighbors(
                    t_graph=g,
                    chunk_sizes=l,
                    start_with_triangles=False,
                    start_from_min=False,
                )
            )
        case "beam_neighbors_triangles":
            vertices = process(
                lambda g, l: _wrong_subgraphs_strategy_beam_neighbors(
                    t_graph=g,
                    chunk_sizes=l,
                    start_with_triangles=True,
                    start_from_min=True,
                )
            )
        case "beam_neighbors_max_triangles":
            vertices = process(
                lambda g, l: _wrong_subgraphs_strategy_beam_neighbors(
                    t_graph=g,
                    chunk_sizes=l,
                    start_with_triangles=True,
                    start_from_min=False,
                )
            )
        case "components_biggest":
            vertices = process(
                lambda g, l: _wrong_subgraphs_strategy_components(
                    g, l, start_from_biggest_component=True
                )
            )
        case "components_spredded":
            vertices = process(
                lambda g, l: _wrong_subgraphs_strategy_components(
                    g, l, start_from_biggest_component=False
                )
            )
        case _:
            raise ValueError(
                f"Unknown strategy: {order_strategy}, supported: none, degree, degree_cycles, cycles, cycles_match_chunks, bfs, beam_neighbors, components_biggest, components_spredded"
            )

    assert vertices_no == len(vertices)

    if NAC_PRINT_SWITCH:
        print("-" * 80)
        print(graphviz_graph(order_strategy, component_to_edges, chunk_sizes, vertices))
        print("-" * 80)
        print(
            graphviz_t_graph(
                t_graph, order_strategy, component_to_edges, chunk_sizes, vertices
            )
        )

        print("-" * 80)
        print(f"Vertices no:  {nx.number_of_nodes(graph)}")
        print(f"Edges no:     {nx.number_of_edges(graph)}")
        print(f"T-graph size: {nx.number_of_nodes(t_graph)}")
        print(f"Comp. to ed.: {component_to_edges}")
        print(f"Chunk no.:    {len(chunk_sizes)}")
        print(f"Chunk sizes:  {chunk_sizes}")
        print("-" * 80)

    @NAC_statistics_colorings_merge_wrapper
    def colorings_merge_wrapper(
        colorings_1: Tuple[Iterable[int], int],
        colorings_2: Tuple[Iterable[int], int],
    ) -> Tuple[Iterable[int], int]:
        (epoch1, subgraph_mask_1) = colorings_1
        (epoch2, subgraph_mask_2) = colorings_2

        epoch1 = RepeatableIterator(epoch1)
        epoch2 = RepeatableIterator(epoch2)
        # epoch2_switched = ( # could be RepeatableIterator
        epoch2_switched = RepeatableIterator(
            # this has to be list so the iterator is not iterated concurrently
            [coloring ^ subgraph_mask_2 for coloring in epoch2]
        )

        return (
            itertools.chain(
                _subgraphs_join_epochs(
                    graph,
                    t_graph,
                    component_to_edges,
                    check_selected_NAC_coloring,
                    from_angle_preserving_components,
                    vertices,
                    epoch1,
                    subgraph_mask_1,
                    epoch2,
                    subgraph_mask_2,
                ),
                _subgraphs_join_epochs(
                    graph,
                    t_graph,
                    component_to_edges,
                    check_selected_NAC_coloring,
                    from_angle_preserving_components,
                    vertices,
                    epoch1,
                    subgraph_mask_1,
                    epoch2_switched,
                    subgraph_mask_2,
                ),
            ),
            subgraph_mask_1 | subgraph_mask_2,
        )

    # Holds all the NAC colorings for a subgraph represented by the second bitmask
    all_epochs: List[Tuple[Iterable[int], int]] = []
    # No. of vertices already processed in previous chunks
    offset = 0
    for chunk_size in chunk_sizes:

        subgraph_mask = 2**chunk_size - 1
        all_epochs.append(
            (
                _subgraph_colorings_generator(
                    graph,
                    t_graph,
                    component_to_edges,
                    check_selected_NAC_coloring,
                    vertices,
                    chunk_size,
                    offset,
                ),
                subgraph_mask << offset,
            )
        )
        offset += chunk_size

    match merge_strategy:
        case "linear":
            res: Tuple[Iterable[int], int] = all_epochs[0]
            for g in all_epochs[1:]:
                res = colorings_merge_wrapper(res, g)
            all_epochs = [res]

        case "sorted_bits":
            """
            Similar to linear, but sorts the subgraphs by size first
            """
            all_epochs = sorted(
                all_epochs, key=lambda x: x[1].bit_count(), reverse=True
            )

            while len(all_epochs) > 1:
                iterable, mask = colorings_merge_wrapper(
                    all_epochs[-1],
                    all_epochs[-2],
                )
                all_epochs.pop()
                all_epochs[-1] = (iterable, mask)

        case "sorted_size":
            """
            Sorts the subgraphs by number of their NAC colorings
            and merges them linearly from the smallest to the biggest
            """
            # Joins the subgraphs like a tree
            all_epochs: List[Tuple[List[int], int]] = [
                (list(i), m) for i, m in all_epochs
            ]

            all_epochs = sorted(all_epochs, key=lambda x: len(x[0]), reverse=True)

            while len(all_epochs) > 1:
                iterable, mask = colorings_merge_wrapper(
                    all_epochs[-1],
                    all_epochs[-2],
                )
                all_epochs.pop()
                all_epochs[-1] = (list(iterable), mask)

        case "log" | "log_reverse":
            # Joins the subgraphs like a tree
            while len(all_epochs) > 1:
                next_all_epochs: List[Tuple[Iterable[int], int]] = []

                # always join 2 subgraphs
                for batch in itertools.batched(all_epochs, 2):
                    if len(batch) == 1:
                        next_all_epochs.append(batch[0])
                        continue

                    next_all_epochs.append(colorings_merge_wrapper(*batch))

                match merge_strategy:
                    case "log_reverse":
                        next_all_epochs = list(reversed(next_all_epochs))

                all_epochs = next_all_epochs

        case "min_max":
            # Joins the subgraphs like a tree
            all_epochs: List[Tuple[List[int], int]] = [
                (list(i), m) for i, m in all_epochs
            ]
            while len(all_epochs) > 1:
                next_all_epochs: List[Tuple[List[int], int]] = []

                all_epochs = sorted(all_epochs, key=lambda x: x[1].bit_count())

                for batch_id in range(len(all_epochs) // 2):

                    iterable, mask = colorings_merge_wrapper(
                        all_epochs[batch_id],
                        all_epochs[-(batch_id + 1)],
                    )
                    next_all_epochs.append((list(iterable), mask))

                if len(all_epochs) % 2 == 1:
                    next_all_epochs.append(all_epochs[len(all_epochs) // 2])

                all_epochs = next_all_epochs

        case "score":
            """
            This approach forbids the online version of the algorithm

            Iterations are run until the original graph is restored.
            In each iteration a score is computed and the pair with
            the best score is chosen.
            Score tries to mimic the work required to join the subgraphs.
            Which is the product of # of colorings on both the subgraphs
            times the size of the resulting subgraph.
            """
            all_epochs: List[Tuple[List[int], int]] = [
                (list(i), m) for i, m in all_epochs
            ]
            while len(all_epochs) > 1:
                best_pair = (0, 1)
                lowers_score = 9_223_372_036_854_775_807
                for i in range(len(all_epochs)):
                    e1 = all_epochs[i]
                    for j in range(i + 1, len(all_epochs)):
                        e2 = all_epochs[j]
                        # size of colorings product * size of graph
                        score = len(e1[0]) * len(e2[0]) * (e1[1] | e2[1]).bit_count()
                        if score < lowers_score:
                            lowers_score = score
                            best_pair = (i, j)

                e1 = all_epochs[best_pair[0]]
                e2 = all_epochs[best_pair[1]]
                # print(f"{len(e1[0])} {len(e2[0])} {(e1[1] | e2[1]).bit_count()}")
                iterable, mask = colorings_merge_wrapper(
                    all_epochs[best_pair[0]],
                    all_epochs[best_pair[1]],
                )
                # this is safe (and slow) as the second coordinate is greater
                all_epochs.pop(best_pair[1])
                all_epochs[best_pair[0]] = (list(iterable), mask)

        case "dynamic":
            # Joins the subgraphs like a tree
            all_epochs: List[Tuple[List[int], int]] = [
                (list(i), m) for i, m in all_epochs
            ]

            all_ones = 2 ** len(all_epochs) - 1
            # print(len(all_epochs))
            # print(all_ones)
            cache: List[None | Tuple[int, int, int, Tuple[int, int]]] = [
                None for _ in range(2 ** len(all_epochs))
            ]
            cache[all_ones] = (0, 0, 0, (0, 1))

            def dynamic_process_search(
                mask: int,
            ) -> Tuple[int, int, int, Tuple[int, int]]:
                if cache[mask] is not None:
                    return cache[mask]

                if mask.bit_count() + 1 == len(all_epochs):
                    index = (mask ^ all_ones).bit_length() - 1
                    a, b = all_epochs[index]
                    return 0, len(a), b.bit_count(), (index, index)

                least_work = 9_223_372_036_854_775_807**2
                result: Tuple[int, int, int, Tuple[int, int]]

                for i in range(len(all_epochs)):
                    if mask & (1 << i):
                        continue

                    e1 = all_epochs[i]
                    for j in range(i + 1, len(all_epochs)):
                        if mask & (1 << j):
                            continue

                        e2 = all_epochs[j]
                        # size of colorings product * size of graph
                        my_size = (e1[1] | e2[1]).bit_count()
                        my_work = len(e1[0]) * len(e2[0]) * my_size
                        my_outputs = len(e1[0]) * len(e2[0])  # TODO heuristics
                        mask ^= (1 << i) | (1 << j)
                        work, outputs, size, _ = dynamic_process_search(mask)
                        final_size = my_size + size
                        final_output = my_outputs * outputs
                        final_work = work + my_work + final_output * final_size
                        mask ^= (1 << i) | (1 << j)
                        if final_work < least_work:
                            least_work = final_work
                            result = (final_work, final_output, final_size, (i, j))
                cache[mask] = result
                return result

            algo_mask = 0
            while algo_mask != all_ones:
                # print(f"{algo_mask=} {bin(algo_mask)} <- {bin(all_ones)}")
                _, _, _, best_pair = dynamic_process_search(algo_mask)
                # print(f"{best_pair=}")

                if best_pair[0] == best_pair[1]:
                    all_epochs = [all_epochs[best_pair[0]]]
                    break

                algo_mask |= 1 << best_pair[1]

                e1 = all_epochs[best_pair[0]]
                e2 = all_epochs[best_pair[1]]
                # print(f"{len(e1[0])} {len(e2[0])} {(e1[1] | e2[1]).bit_count()}")
                iterable, mask = colorings_merge_wrapper(
                    all_epochs[best_pair[0]],
                    all_epochs[best_pair[1]],
                )
                # this is safe (and slow) as the second coordinate is greater
                # all_epochs.pop(best_pair[1])
                all_epochs[best_pair[0]] = (list(iterable), mask)

        case "recursion" | "recursion_almost":
            # Joins the subgraphs like a tree
            all_epochs: List[Tuple[List[int], int]] = [
                (list(i), m) for i, m in all_epochs
            ]

            all_ones = 2 ** len(all_epochs) - 1
            algo_mask = 0

            cache: List[None | Tuple[int, int, int, Tuple[int, int]]] = [
                None for _ in range(2 ** len(all_epochs))
            ]
            cache[all_ones] = (0, 0, 0, (0, 1))

            while algo_mask != all_ones:

                def dynamic_process_search(
                    mask: int,
                ) -> Tuple[int, int, int, Tuple[int, int]]:
                    if cache[mask] is not None:
                        return cache[mask]

                    if mask.bit_count() + 1 == len(all_epochs):
                        index = (mask ^ all_ones).bit_length() - 1
                        a, b = all_epochs[index]
                        return 0, len(a), b.bit_count(), (index, index)

                    least_work = 9_223_372_036_854_775_807**2
                    result: Tuple[int, int, int, Tuple[int, int]]

                    for i in range(len(all_epochs)):
                        if mask & (1 << i):
                            continue

                        e1 = all_epochs[i]
                        for j in range(i + 1, len(all_epochs)):
                            if mask & (1 << j):
                                continue

                            e2 = all_epochs[j]
                            # size of colorings product * size of graph
                            my_size = (e1[1] | e2[1]).bit_count()
                            my_work = len(e1[0]) * len(e2[0]) * my_size
                            my_outputs = len(e1[0]) * len(e2[0])  # TODO heuristics
                            mask ^= (1 << i) | (1 << j)
                            work, outputs, size, _ = dynamic_process_search(mask)
                            final_size = my_size + size
                            final_output = my_outputs * outputs
                            final_work = work + my_work + final_output * final_size
                            mask ^= (1 << i) | (1 << j)
                            if final_work < least_work:
                                least_work = final_work
                                result = (
                                    final_work,
                                    final_output,
                                    final_size,
                                    (i, j),
                                )
                    cache[mask] = result
                    return result

                # actually, the whole cache should be invalidated
                def invalidate_cache(
                    cache: List[None | Tuple[int, int, int, Tuple[int, int]]],
                    bit: int,
                ) -> List[None | Tuple[int, int, int, Tuple[int, int]]]:
                    if merge_strategy == "recursion":
                        cache = [None for _ in range(2 ** len(all_epochs))]
                        cache[all_ones] = (0, 0, 0, (0, 1))
                        return cache

                    mlow = 2**bit - 1
                    mhigh = all_ones ^ mlow
                    for i in range(all_ones // 2):
                        i = ((i & mhigh) << 1) | (mlow & i)
                        cache[i] = None
                    return cache

                _, _, _, best_pair = dynamic_process_search(algo_mask)

                if best_pair[0] == best_pair[1]:
                    all_epochs = [all_epochs[best_pair[0]]]
                    break

                algo_mask |= 1 << best_pair[1]
                cache = invalidate_cache(cache, best_pair[0])

                e1 = all_epochs[best_pair[0]]
                e2 = all_epochs[best_pair[1]]
                # print(f"{len(e1[0])} {len(e2[0])} {(e1[1] | e2[1]).bit_count()}")
                iterable, mask = colorings_merge_wrapper(
                    all_epochs[best_pair[0]],
                    all_epochs[best_pair[1]],
                )
                # this is safe (and slow) as the second coordinate is greater
                # all_epochs.pop(best_pair[1])
                all_epochs[best_pair[0]] = (list(iterable), mask)

        case "shared_vertices":

            def graph_to_vertices(allow_mask: int) -> Set[int]:
                graph_vertices: Set[int] = set()
                for i, edges in enumerate(component_to_edges):
                    address = 1 << i

                    if address & allow_mask == 0:
                        continue

                    for u, v in edges:
                        graph_vertices.add(u)
                        graph_vertices.add(v)
                return graph_vertices

            while len(all_epochs) > 1:
                best = (0, 0, 0)
                subgraph_vertices: List[Set[int]] = [
                    graph_to_vertices(allow_mask) for _, allow_mask in all_epochs
                ]
                for i in range(0, len(subgraph_vertices)):
                    for j in range(i + 1, len(subgraph_vertices)):
                        vert1 = subgraph_vertices[i]
                        vert2 = subgraph_vertices[j]
                        vertex_no = len(vert1.intersection(vert2))
                        if vertex_no > best[0]:
                            best = (vertex_no, i, j)
                res = colorings_merge_wrapper(all_epochs[best[1]], all_epochs[best[2]])
                all_epochs[best[1]] = res
                all_epochs.pop(best[2])
        case _:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    assert len(all_epochs) == 1
    expected_subgraph_mask = 2**vertices_no - 1
    assert expected_subgraph_mask == all_epochs[0][1]

    for mask in all_epochs[0][0]:
        # print(f"Got mask={bin(mask)}")
        if mask == 0 or mask.bit_count() == len(vertices):
            continue

        coloring = _coloring_from_mask(vertices, mask, component_to_edges)

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])


def _NAC_colorings_with_non_surjective(
    comp: nx.Graph, colorings: Iterable[NACColoring]
) -> Iterable[NACColoring]:
    """
    This takes an iterator of NAC colorings and yields from it.
    Before it it sends non-surjective "NAC colorings"
    - all red and all blue coloring.
    """
    r, b = [], list(comp.edges())
    yield r, b
    yield b, r
    yield from colorings


def _NAC_colorings_cross_product(
    first: Iterable[NACColoring], second: Iterable[NACColoring]
) -> Iterable[NACColoring]:
    """
    This makes a cartesian cross product of two NAC coloring iterators.
    Unlike the python crossproduct, this adds the colorings inside the tuples.
    """
    cache = RepeatableIterator(first)
    for s in second:
        for f in cache:
            # yield s[0].extend(f[0]), s[1].extend(f[1])
            yield s[0] + f[0], s[1] + f[1]


def _NAC_colorings_for_single_edges(
    edges: Sequence[Edge], include_non_surjective: bool
) -> Iterable[NACColoring]:
    """
    Creates all the NAC colorings for the edges given.
    The colorings are not check for validity.
    Make sure the edges given cannot form a red/blue almost cycle.
    """

    def binaryToGray(num: int) -> int:
        return num ^ (num >> 1)

    red: Set[Edge] = set()
    blue: Set[Edge] = set(edges)

    if include_non_surjective:
        # create immutable versions
        r, b = list(red), list(blue)
        yield r, b
        yield b, r

    prev_mask = 0
    for mask in range(1, 2 ** len(edges) // 2):
        mask = binaryToGray(mask)
        diff = prev_mask ^ mask
        prev_mask = mask
        is_new = (mask & diff) > 0
        edge = edges[diff.bit_length()]
        if is_new:
            red.add(edge)
            blue.remove(edge)
        else:
            blue.add(edge)
            red.remove(edge)

        # create immutable versions
        r, b = list(red), list(blue)
        yield r, b
        yield b, r


def _NAC_colorings_from_bridges(
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    graph: nx.Graph,
    copy: bool = True,
) -> Iterable[NACColoring]:
    """
    Optimization for NAC coloring search that first finds bridges and
    biconnected components. After that it find coloring for each component
    separately and then combines all the possible found NAC colorings.

    Parameters
    ----------
    processor:
        A function that takes a graph (subgraph of the graph given)
        and finds all the NAC colorings for it.
    copy:
        If the graph should be copied before making any destructive changes.
    ----------

    Returns:
        All the NAC colorings of the graph.
    """

    bridges = list(nx.bridges(graph))
    if len(bridges) == 0:
        return processor(graph)

    if copy:
        graph = nx.Graph(graph)
    graph.remove_edges_from(bridges)
    components = [
        nx.induced_subgraph(graph, comp)
        for comp in nx.components.connected_components(graph)
    ]
    components = filter(lambda x: x.number_of_nodes() > 1, components)

    colorings = [
        _NAC_colorings_with_non_surjective(comp, processor(comp)) for comp in components
    ]
    colorings.append(
        _NAC_colorings_for_single_edges(bridges, include_non_surjective=True),
    )

    iterator = reduce(_NAC_colorings_cross_product, colorings)

    # Skip initial invalid coloring that are not surjective
    iterator = filter(lambda x: len(x[0]) * len(x[1]) != 0, iterator)

    return iterator


def _NAC_colorings_from_articulation_points(
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    graph: nx.Graph,
) -> Iterable[NACColoring]:
    colorings: List[Iterable[NACColoring]] = []
    for component in nx.components.biconnected_components(graph):
        subgraph = nx.induced_subgraph(graph, component)
        iterable = processor(subgraph)
        iterable = _NAC_colorings_with_non_surjective(subgraph, iterable)
        colorings.append(iterable)

    iterator = reduce(_NAC_colorings_cross_product, colorings)

    # Skip initial invalid coloring that are not surjective
    iterator = filter(lambda x: len(x[0]) * len(x[1]) != 0, iterator)

    return iterator


def _NAC_colorings_with_trivial_stable_cuts(
    graph: nx.Graph,
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    copy: bool = True,
) -> Iterable[NACColoring]:
    # TODO don't waste resources
    _, component_to_edge = find_triangle_components(
        graph,
        is_cartesian_NAC_coloring=False,
    )

    verticies_in_triangle_components: Set[int] = set()
    for component_edges in component_to_edge:
        # component is not part of a triangle edge
        if len(component_edges) == 1:
            continue

        verticies_in_triangle_components.update(
            v for edge in component_edges for v in edge
        )

    # All nodes are members of a triangle component
    if len(verticies_in_triangle_components) == graph.number_of_nodes():
        return processor(graph)

    if copy:
        graph = nx.Graph(graph)

    removed_edges: List[Edge] = []
    # vertices outside of triangle components
    for v in verticies_in_triangle_components - set(graph.nodes):
        for u in graph.neighbors(v):
            removed_edges.append((u, v))
        graph.remove_node(v)

    coloring = processor(graph)
    coloring = _NAC_colorings_with_non_surjective(graph, coloring)

    def handle_vertex(
        graph: nx.Graph,
        v: int,
        get_coloring: Callable[[nx.Graph], Iterable[NACColoring]],
    ) -> Iterable[NACColoring]:
        graph = nx.Graph(graph)
        edges: List[Edge] = []
        for u in graph.neighbors(v):
            edges.append((u, v))

        for coloring in get_coloring(graph):
            # create red/blue components
            pass

        pass

    pass


def _renamed_coloring(
    ordered_vertices: Sequence[int],
    colorings: Iterable[NACColoring],
) -> Iterable[NACColoring]:
    """
    Expects graph vertices to be named from 0 to N-1.
    """
    for coloring in colorings:
        yield tuple(
            [(ordered_vertices[u], ordered_vertices[v]) for u, v in group]
            for group in coloring
        )


def _relabel_graph_for_NAC_coloring(
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    graph: nx.Graph,
    seed: int,
    strategy: str = "random",
    copy: bool = True,
    restart_after: int | None = None,
) -> Iterable[NACColoring]:
    vertices = list(graph.nodes)

    if strategy == "none":
        # this is correct, but harmless (for now)
        # return graph if not copy else nx.Graph(graph)
        if set(vertices) == set(range(graph.number_of_nodes())):
            return processor(graph)

        # make all the nodes names in range 0..<n
        mapping = {v: k for k, v in enumerate(vertices)}
        graph = nx.relabel_nodes(graph, mapping, copy=copy)
        return _renamed_coloring(vertices, processor(graph))

    used_vertices: Set[int] = set()
    ordered_vertices: List[int] = []

    if restart_after is None:
        restart_after = graph.number_of_nodes()

    random.Random(seed).shuffle(vertices)

    if strategy == "random":
        mapping = {v: k for k, v in enumerate(vertices)}
        graph = nx.relabel_nodes(graph, mapping, copy=copy)
        return _renamed_coloring(vertices, processor(graph))

    for start in vertices:
        if start in used_vertices:
            continue

        found_vertices = 0
        match strategy:
            case "bfs":
                iterator = nx.bfs_edges(graph, start)
            case "beam_degree":
                iterator = nx.bfs_beam_edges(
                    graph,
                    start,
                    lambda v: nx.degree(graph, v),
                    width=max(5, int(np.sqrt(graph.number_of_nodes()))),
                )
            case _:
                raise ValueError(
                    f"Unknown strategy for relabeling: {strategy}, posible values are none, random, bfs and beam_degree"
                )

        used_vertices.add(start)
        ordered_vertices.append(start)

        for tail, head in iterator:
            if head in used_vertices:
                continue

            used_vertices.add(head)
            ordered_vertices.append(head)

            found_vertices += 1
            if found_vertices >= restart_after:
                break

        assert start in used_vertices

    mapping = {v: k for k, v in enumerate(ordered_vertices)}
    graph = nx.relabel_nodes(graph, mapping, copy=copy)
    return _renamed_coloring(ordered_vertices, processor(graph))


def _NAC_colorings_without_vertex(
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    graph: nx.Graph,
    vertex: int | None,
) -> Iterable[NACColoring]:

    if graph.number_of_nodes() <= 1:
        return

    if vertex is None:
        vertex = min(graph.degree, key=lambda vd: vd[1])[0]

    subgraph = nx.Graph(graph)
    subgraph.remove_node(vertex)

    endpoints = list(graph.neighbors(vertex))
    # print()
    # print(Graph(graph))
    # print(Graph(subgraph))

    # TODO cache when the last coloring is opposite of the current one
    iterator = iter(
        _NAC_colorings_with_non_surjective(
            subgraph,
            processor(subgraph),
        )
    )

    # In case there are no edges in the subgraph,
    # both the all red and blue components are empty
    if subgraph.number_of_edges() == 0:
        r, b = next(iterator)
        assert len(r) == 0
        assert len(b) == 0

    # in many of my algorithms I return coloring and it's symmetric
    # variant after each other. In that case local results are also symmetric
    previous_coloring: NACColoring | None = None
    previous_results: List[NACColoring] = []

    for coloring in iterator:
        if previous_coloring is not None:
            if (
                coloring[0] == previous_coloring[1]
                and coloring[1] == previous_coloring[0]
            ):
                for previous in previous_results:
                    yield previous[1], previous[0]
                previous_coloring = None
                continue
        previous_coloring = coloring
        previous_results = []

        # Naming explanation
        # vertex -> forbidden
        # endpoint -> for each edge (vertex, u), it's u
        # edge -> one of the edges incident to vertex
        # component -> connected component in the coloring given
        # group -> set of components
        #          if there are more edges/endpoints, that share the same color

        # print(f"{coloring=}")

        # create red & blue components
        def find_components_and_neighbors(
            red_edges: Collection[Edge],
            blue_edges: Collection[Edge],
        ) -> Tuple[Dict[int, int], List[Set[int]]]:
            sub = nx.Graph()
            sub.add_nodes_from(endpoints)
            sub.add_edges_from(red_edges)
            vertex_to_comp: Dict[int, int] = defaultdict(lambda: -1)
            id = -1
            for id, comp in enumerate(nx.connected_components(sub)):
                for v in comp:
                    vertex_to_comp[v] = id
            neighboring_components: List[Set[int]] = [set() for _ in range(id + 1)]
            for u, v in blue_edges:
                c1, c2 = vertex_to_comp[u], vertex_to_comp[v]
                if c1 != c2 and c1 != -1 and c2 != -1:
                    neighboring_components[c1].add(c2)
                    neighboring_components[c2].add(c1)
            return vertex_to_comp, neighboring_components

        red_endpoint_to_comp, red_neighboring_components = (
            find_components_and_neighbors(coloring[0], coloring[1])
        )
        blue_endpoint_to_comp_id, blue_neighboring_components = (
            find_components_and_neighbors(coloring[1], coloring[0])
        )
        endpoint_to_comp_id = (red_endpoint_to_comp, blue_endpoint_to_comp_id)
        neighboring_components = (
            red_neighboring_components,
            blue_neighboring_components,
        )
        # print(f"{endpoint_to_comp_id=}")
        # print(f"{neighboring_components=}")

        # find endpoints that are connected to the same component
        # we do this for both the red and blue, as this constrain has
        # to be fulfilled for both of them
        comp_to_endpoint: Tuple[Dict[int, int], Dict[int, int]] = ({}, {})
        same_groups = UnionFind()
        for u in endpoints:
            comp_id = endpoint_to_comp_id[0][u]
            if comp_id < 0:
                pass
            elif comp_id in comp_to_endpoint[0]:
                # if there is already a vertex for this component, we merge
                same_groups.join(comp_to_endpoint[0][comp_id], u)
            else:
                # otherwise we create a first record for the edge
                comp_to_endpoint[0][comp_id] = u

            # same as above, but for blue
            comp_id = endpoint_to_comp_id[1][u]
            if comp_id < 0:
                pass
            elif comp_id in comp_to_endpoint[1]:
                same_groups.join(comp_to_endpoint[1][comp_id], u)
            else:
                comp_to_endpoint[1][comp_id] = u

        # print(f"{same_groups=}")
        # print(f"{comp_to_endpoint=}")

        # Now we need to create an equivalent of component_to_edges
        # or to be precise vertex to edges
        # endpoint_to_same: List[List[int]] = [[] for _ in endpoints]
        endpoint_to_same: Dict[int, List[int]] = defaultdict(list)
        unique = set()
        for u in endpoints:
            id = same_groups.find(u)
            endpoint_to_same[id].append(u)
            unique.add(id)
        # print(f"{endpoint_to_same=}")

        # Now we need to remove empty components,
        # but keep relation to neighbors (update vertex_to_components)
        groups: List[List[int]] = []
        group_to_components: Tuple[List[List[int]], List[List[int]]] = (
            [],
            [],
        )  # holds component ids
        group_to_components_masks: Tuple[List[int], List[int]] = (
            [],
            [],
        )  # holds component ids

        for endpoints_in_same_group in endpoint_to_same.values():
            if len(endpoints_in_same_group) == 0:
                continue

            groups.append(endpoints_in_same_group)
            group_to_components[0].append(
                list(
                    filter(
                        lambda x: x >= 0,
                        (endpoint_to_comp_id[0][u] for u in endpoints_in_same_group),
                    )
                )
            )
            group_to_components[1].append(
                list(
                    filter(
                        lambda x: x >= 0,
                        (endpoint_to_comp_id[1][u] for u in endpoints_in_same_group),
                    )
                )
            )
            mask0, mask1 = 0, 0
            for u in endpoints_in_same_group:
                if u >= 0:
                    mask0 |= 1 << endpoint_to_comp_id[0][u]
                    mask1 |= 1 << endpoint_to_comp_id[1][u]
            group_to_components_masks[0].append(mask0)
            group_to_components_masks[1].append(mask1)

        # print(f"{groups=}")
        # print(f"{group_to_components=}")

        # Map everything to groups so we don't need to use indirection later
        groups_neighbors: Tuple[List[int], List[int]] = ([], [])
        for red_components, blue_components in zip(*group_to_components):
            # groups_neighbors[0].append(
            #     {n for comp_id in red_components for n in neighboring_components[0][comp_id]}
            # )
            # groups_neighbors[1].append(
            #     {n for comp_id in blue_components for n in neighboring_components[1][comp_id]}
            # )

            mask = 0
            for comp_id in red_components:
                for n in neighboring_components[0][comp_id]:
                    mask |= 1 << n
            groups_neighbors[0].append(mask)

            mask = 0
            for comp_id in blue_components:
                for n in neighboring_components[1][comp_id]:
                    mask |= 1 << n
            groups_neighbors[1].append(mask)

        # print(f"{groups_neighbors=}")

        # Check for contradictions
        # if there are in one component edges that share the same component
        # using one color, but use neighboring components using the other
        # color, we can exit now as all the coloring cannot pass
        # contradiction_found = False
        # TODO not working
        # for red_components, blue_components, red_neighbors, blue_neighbors in zip(
        #     *group_to_components, *groups_neighbors
        # ):
        #     if len(red_neighbors.intersection(red_components)) > 0:
        #         contradiction_found = True
        #     if len(blue_neighbors.intersection(blue_components)) > 0:
        #         contradiction_found = True
        # if contradiction_found:
        #     print("Contradiction found")
        #     break
        # TODO check if not all the components are neighboring

        groups_edges: List[List[Edge]] = [
            [(vertex, endpoint) for endpoint in group] for group in groups
        ]

        # Now we iterate all the possible colorings of groups
        for mask in range(2 ** len(groups)):
            # we create red and blue components lists
            # current_comps = ([], [])
            # for i in range(len(groups)):
            #     if mask & (1 << i):
            #         current_comps[0].extend(group_to_components[0][i])
            #     else:
            #         current_comps[1].extend(group_to_components[1][i])
            current_comps0, current_comps1 = (0, 0)
            for i in range(len(groups)):
                if mask & (1 << i):
                    current_comps0 |= group_to_components_masks[0][i]
                else:
                    current_comps1 |= group_to_components_masks[1][i]

            # no neighboring connected by the same color
            r = iter(range(len(groups)))
            i = next(r, None)
            valid = True
            while valid and i is not None:
                if mask & (1 << i):
                    if groups_neighbors[0][i] & current_comps0:
                        valid = False
                else:
                    if groups_neighbors[1][i] & current_comps1:
                        valid = False
                i = next(r, None)

            if not valid:
                continue

            current_edges: Tuple[List[Edge], List[Edge]] = ([], [])
            for i in range(len(groups)):
                k = int((mask & (1 << i)) == 0)
                current_edges[k].extend(groups_edges[i])

            to_emit = (
                coloring[0] + current_edges[0],
                coloring[1] + current_edges[1],
            )
            if len(to_emit[0]) * len(to_emit[1]) > 0:
                # print(f"{graph.number_of_nodes()} -> {to_emit=}")
                # assert nx.Graph(graph).is_NAC_coloring(to_emit)

                yield to_emit
                previous_results.append(to_emit)
        # print()


def NAC_colorings_impl(
    self: nx.Graph,
    algorithm: str,
    relabel_strategy: str,
    use_decompositions: bool,
    is_cartesian: bool,
    use_chromatic_partitions: bool,
    remove_vertices_cnt: int,
    use_has_coloring_check: bool,  # I disable the check in tests
    seed: int | None,
) -> Iterable[NACColoring]:
    if not check_NAC_constrains(self):
        return []

    # Checks if it even makes sense to do the search
    if use_has_coloring_check and has_NAC_coloring_checks(self) == False:
        return []

    rand = random.Random(seed)

    def run(graph: nx.Graph) -> Iterable[NACColoring]:
        # TODO NAC not sure if copy is needed, but I used it before
        graph = NiceGraph(graph)

        # in case graph has no edges because of some previous optimizations,
        # there are no NAC colorings
        if graph.number_of_edges() == 0:
            return []

        if use_chromatic_partitions:
            edge_to_component, component_to_edge = find_triangle_components(
                graph,
                is_cartesian_NAC_coloring=is_cartesian,
            )
        else:
            edge_to_component, component_to_edge = fake_triangle_components(graph)
        t_graph = create_T_graph_from_components(graph, edge_to_component)

        algorithm_parts = list(algorithm.split("-"))
        match algorithm_parts[0]:
            case "naive":
                is_NAC_coloring = (
                    _is_cartesian_NAC_coloring_impl
                    if is_cartesian
                    else _is_NAC_coloring_impl
                )
                return _NAC_colorings_naive(
                    graph,
                    t_graph,
                    component_to_edge,
                    is_NAC_coloring,
                )
            case "cycles":
                is_NAC_coloring = (
                    _is_cartesian_NAC_coloring_impl
                    if is_cartesian
                    else _is_NAC_coloring_impl
                )

                if len(algorithm_parts) == 1:
                    return _NAC_colorings_cycles(
                        graph,
                        t_graph,
                        component_to_edge,
                        is_NAC_coloring,
                        is_cartesian,
                    )
                return _NAC_colorings_cycles(
                    graph,
                    t_graph,
                    component_to_edge,
                    is_NAC_coloring,
                    from_angle_preserving_components=is_cartesian,
                    use_all_cycles=bool(algorithm_parts[1]),
                )
            case "subgraphs":
                is_NAC_coloring = (
                    _is_cartesian_NAC_coloring_impl
                    if is_cartesian
                    else _is_NAC_coloring_impl
                )

                if len(algorithm_parts) == 1:
                    return _NAC_colorings_subgraphs(
                        graph,
                        t_graph,
                        component_to_edge,
                        is_NAC_coloring,
                        from_angle_preserving_components=is_cartesian,
                        seed=rand.randint(0, 2**16 - 1),
                    )
                return _NAC_colorings_subgraphs(
                    graph,
                    t_graph,
                    component_to_edge,
                    is_NAC_coloring,
                    from_angle_preserving_components=is_cartesian,
                    seed=rand.randint(0, 2**16 - 1),
                    merge_strategy=algorithm_parts[1],
                    order_strategy=algorithm_parts[2],
                    preferred_chunk_size=(
                        int(algorithm_parts[3])
                        if algorithm_parts[3] != "auto"
                        else None
                    ),
                    get_subgraphs_together=(
                        algorithm_parts[4] == "smart"
                        if len(algorithm_parts) == 5
                        else False
                    ),
                )
            case _:
                raise ValueError(f"Unknown algorighm type: {algorithm}")

    def apply_processor(
        processor: Callable[[nx.Graph], Iterable[NACColoring]],
        func: Callable[
            [Callable[[nx.Graph], Iterable[NACColoring]], nx.Graph],
            Iterable[NACColoring],
        ],
    ) -> Callable[[nx.Graph], Iterable[NACColoring]]:
        return lambda g: func(processor, g)

    graph: nx.Graph = self
    processor: Callable[[nx.Graph], Iterable[NACColoring]] = run

    assert not (is_cartesian and remove_vertices_cnt > 0)
    for _ in range(remove_vertices_cnt):
        processor = apply_processor(
            processor, lambda p, g: _NAC_colorings_without_vertex(p, g, None)
        )

    if use_decompositions:
        # processor = apply_processor(
        #     processor, lambda p, g: _NAC_colorings_from_bridges(p, g)
        # )
        processor = apply_processor(
            processor,
            lambda p, g: _NAC_colorings_from_articulation_points(p, g),
        )

    processor = apply_processor(
        processor,
        lambda p, g: _relabel_graph_for_NAC_coloring(
            p, g, strategy=relabel_strategy, seed=rand.randint(0, 2**16 - 1)
        ),
    )

    return processor(graph)


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
