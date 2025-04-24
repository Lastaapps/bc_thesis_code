from typing import NamedTuple
import networkx as nx

from pyrigi.data_type import Vertex


class SeparatingCut[T: Vertex](NamedTuple):
    """
    Represents a separating cut in a graph.

    Members
    -------
    a:
        vertices in the first component **excluding** the cut vertices
    b:
        vertices in the second component **excluding** the cut vertices
    cut:
        vertices of the cut
    """

    a: set[T]
    b: set[T]
    cut: set[T]

    def __repr__(self) -> str:
        return f"SeparatingCut({self.a}, {self.b} - {self.cut})"

    def __eq__(self, other) -> bool:
        if self.cut != other.cut:
            return False
        if self.a == other.a:
            return self.b == other.b
        return self.a == other.b and self.b == other.a


class StableCut[T: Vertex](SeparatingCut[T]):
    """
    Represents a stable cutset in a graph.

    Members
    -------
    a:
        vertices in the first component **excluding** the cut vertices
    b:
        vertices in the second component **excluding** the cut vertices
    cut:
        vertices of the cut
    """

    def __repr__(self) -> str:
        return f"StableCut({self.a}, {self.b} - {self.cut})"

    def validate(self, graph: nx.Graph) -> bool:
        """
        Checks if the this cut is a stable cut of the given graph

        Parameters
        ----------

        """
        from stablecut.util import is_stable_cutset

        return is_stable_cutset(graph, self)
