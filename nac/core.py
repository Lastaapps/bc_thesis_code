from collections import defaultdict, deque
from queue import PriorityQueue
from functools import reduce
import itertools
import random
from typing import *

import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
import math
import numpy as np

from nac.util.union_find import UnionFind
from nac.util.repetable_iterator import RepeatableIterator

from nac.data_type import NACColoring, Edge
from nac.monochromatic_classes import (
    MonochromaticClassType,
    find_monochromatic_classes,
    create_T_graph_from_components,
)
from nac.existence import has_NAC_coloring_checks, check_NAC_constrains
from nac.check import (
    _is_cartesian_NAC_coloring_impl,
    _is_NAC_coloring_impl,
    _NAC_check_called_reset,
)
from nac.development import (
    NAC_PRINT_SWITCH,
    NAC_statistics_generator,
    NAC_statistics_colorings_merge_wrapper,
    graphviz_graph,
    graphviz_t_graph,
)
from nac.util import NiceGraph

from nac.cycle_detection import find_cycles
import nac.check


def coloring_from_mask(
    ordered_comp_ids: List[int],
    component_to_edges: List[List[Edge]],
    mask: int,
    allow_mask: int | None = None,
) -> NACColoring:
    """
    Converts a mask representing a red-blue edge coloring.

    Parameters
    ----------
    ordered_comp_ids:
        list of component ids, mask points into it
    component_to_edges:
        mapping from component id to its edges
    mask:
        bit mask pointing into ordered_comp_ids,
        1 means red and 0 blue (or otherwise)
    allow_mask:
        mask allowing only some components.
        Used when generating coloring for subgraph.
    """

    if allow_mask is None:
        allow_mask = 2 ** len(ordered_comp_ids) - 1

    red, blue = [], []  # set(), set()
    for i, e in enumerate(ordered_comp_ids):
        address = 1 << i

        if address & allow_mask == 0:
            continue

        edges = component_to_edges[e]
        # (red if mask & address else blue).update(edges)
        (red if mask & address else blue).extend(edges)
    return (red, blue)

    # numpy impl, ~10% slower
    # if allow_mask is not None:
    #     ordered_comp_ids = ordered_comp_ids[allow_mask]
    #     mask = mask[allow_mask]

    # red_vert = ordered_comp_ids[mask]
    # blue_vert = ordered_comp_ids[~mask]

    # red = [edge for edges in red_vert for edge in component_to_edges[edges]]
    # blue = [edge for edges in blue_vert for edge in component_to_edges[edges]]

    # return (red, blue)


################################################################################
def create_bitmask_for_t_graph_cycle(
    graph: nx.Graph,
    component_to_edges: Callable[[int], List[Edge]],
    cycle: Tuple[int, ...],
    local_ordered_comp_ids: Set[int] | None = None,
) -> Tuple[int, int]:
    """
    Creates a bit mask (template) to match components in the cycle
    and a mask matching components of the cycle that make NAC coloring
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
    # template = np.zeros(len(ordered_comp_ids), dtype=np.bool)
    # valid = np.zeros(len(ordered_comp_id), dtype=np.bool)

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

        if local_ordered_comp_ids is not None:
            intersections_prev = intersections_prev.intersection(local_ordered_comp_ids)
            intersections_next = intersections_next.intersection(local_ordered_comp_ids)

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


def mask_matches_templates(
    templates: List[Tuple[int, int]],
    mask: int,
    subgraph_mask: int,
) -> bool:
    """
    Checks if mask given matches any of the cycles given.

    Parameters
    ----------
        templates:
            list of outputs of the create_bitmask_for_t_graph_cycle
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
    nac.check._NAC_CHECK_CYCLE_MASK += 1

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
