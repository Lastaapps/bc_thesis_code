from typing import NamedTuple
import networkx as nx

from pyrigi.data_type import Vertex


class VertexCut[T: Vertex](NamedTuple):
    """
    Represents a cut in a graph

    Fileds
    ------
    a     vertices in the first component **excluding** the cut vertices
    b     vertices in the second component **excluding** the cut vertices
    cut   vertices of the cut
    """

    a: set[T]
    b: set[T]
    cut: set[T]

    def validate(self, graph: nx.Graph) -> bool:
        from stablecut.util import is_stable_set

        a, b, c = self
        return (
            len(a & b) == 0
            and len(a | b | c) == graph.number_of_nodes()
            and is_stable_set(graph, c)
        )

    def __repr__(self) -> str:
        return f"VertexCut({self.a}, {self.b} - {self.cut})"

    def __eq__(self, other) -> bool:
        if self.cut != other.cut:
            return False
        if self.a != other.a:
            return self.a == other.b and self.b == other.a
        return self.a == other.b


class StableCut[T: Vertex](VertexCut[T]):
    pass
