"""
This module handles all the variants of cycle detection
"""

from collections import defaultdict, deque
from typing import *

import networkx as nx

from pyrigi.data_type import Edge


# obsolete, left for legacy reasons
def _find_cycles_in_component_graph(
    graph: nx.Graph,
    comp_graph: nx.Graph,
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

    vertices = list(comp_graph.nodes)

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

        for u in comp_graph.neighbors(start):
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

            for u in comp_graph.neighbors(v):
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


def find_cycles(
    graph: nx.Graph,
    subgraph_components: Set[int],
    component_to_edges: List[Set[Tuple[int, int]]],
    all: bool = False,
) -> Set[Tuple[int, ...]]:
    match 2:
        case 0:
            return set()
        case 1:
            return _find_shortest_cycles(
                graph, subgraph_components, component_to_edges, all
            )
        case 2:
            return _find_useful_cycles(graph, subgraph_components, component_to_edges)
        case 3:
            res = _find_shortest_cycles(
                graph, subgraph_components, component_to_edges, all
            ) | _find_useful_cycles(graph, subgraph_components, component_to_edges)
            res = list(sorted(res, key=lambda x: len(x)))[: 2 * graph.number_of_nodes()]
            return set(res)


def _find_shortest_cycles(
    graph: nx.Graph,
    subgraph_components: Set[int],
    component_to_edges: List[Set[Tuple[int, int]]],
    all: bool = False,
    per_class_limit: int = 1024,
) -> Set[Tuple[int, ...]]:
    cycles = _find_shortest_cycles_for_components(
        graph=graph,
        subgraph_components=subgraph_components,
        component_to_edges=component_to_edges,
        all=all,
        per_class_limit=per_class_limit,
    )
    return {c for comp_cycles in cycles.values() for c in comp_cycles}


def _find_shortest_cycles_for_components(
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


def _find_useful_cycles(
    graph: nx.Graph,
    subgraph_components: Set[int],
    component_to_edges: List[Set[Tuple[int, int]]],
    per_class_limit: int = 2,
) -> Set[Tuple[int, ...]]:
    cycles = _find_useful_cycles_for_components(
        graph=graph,
        subgraph_components=subgraph_components,
        component_to_edges=component_to_edges,
        per_class_limit=per_class_limit,
    )
    return {c for comp_cycles in cycles.values() for c in comp_cycles}


def _find_useful_cycles_for_components(
    graph: nx.Graph,
    subgraph_components: Set[int],
    component_to_edges: List[Set[Tuple[int, int]]],
    per_class_limit: int = 2,
) -> Dict[int, Set[Tuple[int, ...]]]:
    """
    For each edge finds all the cycles among monochromatic classes
    of length at most five (length is the number of classes it spans).

    Not all the returned cycles are guaranteed to be actual cycles as
    this may create cycles that enter and exit a component at the same vertex.
    """
    comp_no = len(component_to_edges)

    # creates mapping from vertex to set of monochromatic classes if is in
    vertex_to_components = [set() for _ in range(max(graph.nodes) + 1)]
    for comp_id, comp in enumerate(component_to_edges):
        if comp_id not in subgraph_components:
            continue
        for u, v in comp:
            vertex_to_components[u].add(comp_id)
            vertex_to_components[v].add(comp_id)
    neighboring_components = [set() for _ in range(comp_no)]

    found_cycles: Dict[int, Set[Tuple[int, ...]]] = defaultdict(set)

    # create a graph where vertices are monochromatic classes an there's
    # an edge if the monochromatic classes share a vertex
    for v in graph.nodes:
        for i in vertex_to_components[v]:
            for j in vertex_to_components[v]:
                if i != j:
                    neighboring_components[i].add(j)

    def insert_cycle(comp_id: int, cycle: Tuple[int, ...]):
        """
        Makes sure a cycles is inserted in canonical form to prevent
        having the same cycle more times.

        The canonical form is that the first id is the smallest component
        number in the cycle and the second one is the lower of the neighbors
        in the cycle.
        """
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
        u_comps = vertex_to_components[u]
        v_comps = vertex_to_components[v]

        # remove shared components
        intersection = u_comps.intersection(v_comps)
        u_comps = u_comps - intersection
        v_comps = v_comps - intersection
        # TODO reenable - this only makes sense for proper monochromatic classes, triangle components fail on it
        # assert len(intersection) <= 1
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
