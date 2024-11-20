from typing import *

from nac.data_type import NACColoring, Edge


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
