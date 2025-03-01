from typing import Hashable, NamedTuple
import networkx as nx

from stablecut.util import is_stable_set


# V - vertex type
class StableCut[V: Hashable](NamedTuple):
    a: set[V]
    b: set[V]
    cut: set[V]

    def validate(self, graph: nx.Graph) -> bool:
        a, b, c = self
        return (
            a & b == c
            and len(a) + len(b) - len(c) == graph.number_of_nodes()
            and is_stable_set(graph, c)
        )
