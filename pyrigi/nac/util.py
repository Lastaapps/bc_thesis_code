from typing import *

import networkx as nx

from pyrigi.nac.data_type import NACColoring

from pyrigi.nac.data_type import Edge


def canonical_NAC_coloring(
    coloring: NACColoring,
    including_red_blue_order: bool = True,
) -> NACColoring:
    """
    Expects graph vertices to be comparable.

    Converts coloring into a canonical form by sorting edges,
    vertices in the edges and red/blue part.
    Useful for (equivalence) testing.

    The returned type is guaranteed to be Hashable and Comparable.
    """

    def process(data: Collection[Edge]) -> Tuple[Edge, ...]:
        return tuple(sorted([tuple(sorted(edge)) for edge in data]))

    red, blue = process(coloring[0]), process(coloring[1])

    if including_red_blue_order and red > blue:
        red, blue = blue, red

    return red, blue


class NiceGraph(nx.Graph):
    def __str__(self) -> str:
        return graph_str(self)


def graph_str(graph: nx.Graph) -> str:
    return f"Graph (|V|={graph.number_of_nodes()},|E|={graph.number_of_edges()}) ({list(graph.nodes)} {list(graph.edges)})"
