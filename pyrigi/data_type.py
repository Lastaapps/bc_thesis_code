"""

Module for defining data type used for type hinting.

"""

from typing import Collection, Set, TypeVar, List, Tuple, Hashable


Vertex = Hashable
"""
Any hashable type can be used for a Vertex.
"""

Edge = Tuple[Vertex, Vertex]
"""
An Edge is a pair of :obj:`Vertices <pyrigi.data_type.Vertex>`.
"""

Point = List[float]
"""
A Point is a list of coordinates whose length is the dimension of its affine space.
"""

GraphType = TypeVar("Graph")
FrameworkType = TypeVar("Framework")
MatroidType = TypeVar("Matroid")

NACColoring = Tuple[Collection[Edge], Collection[Edge]]
