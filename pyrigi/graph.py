"""
Module for rigidity related graph properties.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from itertools import combinations
from typing import Iterable, List, Any, Union, Tuple, Optional, Dict, Set
from enum import Enum

import networkx as nx
from sympy import Matrix
import math

from pyrigi.datastructures.union_find import UnionFind
from pyrigi.data_type import NACColoring, Vertex, Edge, GraphType, FrameworkType
from pyrigi.misc import doc_category, generate_category_tables
from pyrigi.exception import LoopError


class Graph(nx.Graph):
    """
    Class representing a graph.

    One option for *incoming_graph_data* is a list of edges.
    See :class:`networkx.Graph` for the other input formats
    or use class methods :meth:`~Graph.from_vertices_and_edges`
    or :meth:`~Graph.from_vertices` when specifying the vertex set is needed.

    Examples
    --------
    >>> from pyrigi import Graph
    >>> G = Graph([(0,1), (1,2), (2,3), (0,3)])
    >>> print(G)
    Graph with vertices [0, 1, 2, 3] and edges [[0, 1], [0, 3], [1, 2], [2, 3]]

    >>> G = Graph()
    >>> G.add_vertices([0,2,5,7,'a'])
    >>> G.add_edges([(0,7), (2,5)])
    >>> print(G)
    Graph with vertices [0, 2, 5, 7, 'a'] and edges [[0, 7], [2, 5]]

    TODO
    ----
    Implement an alias for plotting.
    Graphical output in Jupyter.
    Graph names.

    METHODS

    Notes
    -----
    This class inherits the class :class:`networkx.Graph`.
    Some of the inherited methods are for instance:

    .. autosummary::

        networkx.Graph.add_edge

    Many of the :doc:`NetworkX <networkx:index>` algorithms are implemented as functions,
    namely, a :class:`Graph` instance has to be passed as the first parameter.
    See for instance:

    .. autosummary::

        ~networkx.classes.function.degree
        ~networkx.classes.function.neighbors
        ~networkx.classes.function.non_neighbors
        ~networkx.classes.function.subgraph
        ~networkx.classes.function.edge_subgraph
        ~networkx.classes.function.edges
        ~networkx.algorithms.connectivity.edge_augmentation.is_k_edge_connected
        ~networkx.algorithms.components.is_connected
        ~networkx.algorithms.tree.recognition.is_tree

    The following links give more information on :class:`networkx.Graph` functionality:

    - :doc:`Graph display <networkx:reference/drawing>`
    - :doc:`Directed Graphs <networkx:reference/classes/digraph>`
    - :doc:`Linear Algebra on Graphs <networkx:reference/linalg>`
    - :doc:`A Database of some Graphs <networkx:reference/generators>`
    - :doc:`Reading and Writing Graphs <networkx:reference/readwrite/index>`
    - :doc:`Converting to and from other Data Formats <networkx:reference/convert>`
    """

    def __str__(self) -> str:
        """
        Return the string representation.
        """
        return (
            self.__class__.__name__
            + f" with vertices {self.vertex_list()} and edges {self.edge_list()}"
        )

    def __repr__(self) -> str:
        """
        Return a representation.
        """
        return self.__str__()

    @classmethod
    @doc_category("Class methods")
    def from_vertices_and_edges(
        cls, vertices: List[Vertex], edges: List[Edge]
    ) -> GraphType:
        """
        Create a graph from a list of vertices and edges.

        Parameters
        ----------
        vertices
        edges:
            Edges are tuples of vertices. They can either be a tuple ``(i,j)`` or
            a list ``[i,j]`` with two entries.

        TODO
        ----
        examples, tests
        """
        G = Graph()
        G.add_nodes_from(vertices)
        for edge in edges:
            if len(edge) != 2 or not edge[0] in G.nodes or not edge[1] in G.nodes:
                raise TypeError(
                    f"Edge {edge} does not have the correct format "
                    "or has adjacent vertices the graph does not contain"
                )
            G.add_edge(*edge)
        return G

    @classmethod
    @doc_category("Class methods")
    def from_vertices(cls, vertices: List[Vertex]) -> GraphType:
        """
        Create a graph with no edges from a list of vertices.

        Examples
        --------
        >>> from pyrigi import Graph
        >>> G = Graph.from_vertices([3, 1, 7, 2, 12, 3, 0])
        >>> G
        Graph with vertices [0, 1, 2, 3, 7, 12] and edges []
        """
        return Graph.from_vertices_and_edges(vertices, [])

    @classmethod
    @doc_category("Class methods")
    def CompleteOnVertices(cls, vertices: List[Vertex]) -> GraphType:
        """
        Generate a complete graph on ``vertices``.

        TODO
        ----
        examples, tests
        """
        edges = combinations(vertices, 2)
        return Graph.from_vertices_and_edges(vertices, edges)

    @doc_category("Attribute getters")
    def vertex_list(self) -> List[Vertex]:
        """
        Return the list of vertices.

        The output is sorted if possible,
        otherwise, the internal order is used instead.

        TODO
        ----
        examples
        """
        try:
            return sorted(self.nodes)
        except BaseException:
            return list(self.nodes)

    @doc_category("Attribute getters")
    def edge_list(self) -> List[Edge]:
        """
        Return the list of edges.

        The output is sorted if possible,
        otherwise, the internal order is used instead.

        TODO
        ----
        examples
        """
        try:
            return sorted([sorted(e) for e in self.edges])
        except BaseException:
            return list(self.edges)

    @doc_category("Graph manipulation")
    def delete_vertex(self, vertex: Vertex) -> None:
        """Alias for :meth:`networkx.Graph.remove_node`."""
        self.remove_node(vertex)

    @doc_category("Graph manipulation")
    def delete_vertices(self, vertices: Iterable[Vertex]) -> None:
        """Alias for :meth:`networkx.Graph.remove_nodes_from`."""
        self.remove_nodes_from(vertices)

    @doc_category("Graph manipulation")
    def delete_edge(self, edge: Edge) -> None:
        """Alias for :meth:`networkx.Graph.remove_edge`"""
        self.remove_edge(*edge)

    @doc_category("Graph manipulation")
    def delete_edges(self, edges: Iterable[Edge]) -> None:
        """Alias for :meth:`networkx.Graph.remove_edges_from`."""
        self.remove_edges_from(edges)

    @doc_category("Graph manipulation")
    def add_vertex(self, vertex: Vertex) -> None:
        """Alias for :meth:`networkx.Graph.add_node`."""
        self.add_node(vertex)

    @doc_category("Graph manipulation")
    def add_vertices(self, vertices: Iterable[Vertex]) -> None:
        """Alias for :meth:`networkx.Graph.add_nodes_from`."""
        self.add_nodes_from(vertices)

    @doc_category("Graph manipulation")
    def add_edges(self, edges: Iterable[Edge]) -> None:
        """Alias for :meth:`networkx.Graph.add_edges_from`."""
        self.add_edges_from(edges)

    @doc_category("Graph manipulation")
    def delete_loops(self) -> None:
        """Removes all the loops from the edges to get a loop free graph."""
        self.delete_edges(nx.selfloop_edges(self))

    @doc_category("General graph theoretical properties")
    def vertex_connectivity(self) -> int:
        """Alias for :func:`networkx.algorithms.connectivity.connectivity.node_connectivity`."""  # noqa: E501
        return nx.node_connectivity(self)

    @doc_category("General graph theoretical properties")
    def degree_sequence(self, vertex_order: List[Vertex] = None) -> list[int]:
        """
        Return a list of degrees of the vertices of the graph.

        Parameters
        ----------
        vertex_order:
            By listing vertices in the preferred order, the degree_sequence
            can be computed in a way the user expects. If no vertex order is
            provided, :meth:`~.Graph.vertex_list()` is used.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G.degree_sequence()
        [1, 2, 1]
        """
        if vertex_order is None:
            vertex_order = self.vertex_list()
        else:
            if not set(self.nodes) == set(
                vertex_order
            ) or not self.number_of_nodes() == len(vertex_order):
                raise IndexError(
                    "The vertex_order must contain the same vertices as the graph!"
                )
        return [self.degree(v) for v in vertex_order]

    @doc_category("General graph theoretical properties")
    def min_degree(self) -> int:
        """
        Return the minimum of the vertex degrees.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G.min_degree()
        1
        """
        return min([self.degree(v) for v in self.nodes])

    @doc_category("General graph theoretical properties")
    def max_degree(self) -> int:
        """
        Return the maximum of the vertex degrees.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G.max_degree()
        2
        """
        return max([self.degree(v) for v in self.nodes])

    @doc_category("Sparseness")
    def is_sparse(self, K: int, L: int) -> bool:
        r"""
        Check whether the graph is :prf:ref:`(K, L)-sparse <def-kl-sparse-tight>`.

        TODO
        ----
        pebble game algorithm, examples, tests for other cases than (2,3)
        """
        if not (isinstance(K, int) and isinstance(L, int)):
            raise TypeError("K and L need to be integers!")

        for j in range(K, self.number_of_nodes() + 1):
            for vertex_set in combinations(self.nodes, j):
                G = self.subgraph(vertex_set)
                if G.number_of_edges() > K * G.number_of_nodes() - L:
                    return False
        return True

    @doc_category("Sparseness")
    def is_tight(self, K: int, L: int) -> bool:
        r"""
        Check whether the graph is :prf:ref:`(K, L)-tight <def-kl-sparse-tight>`.

        TODO
        ----
        examples, tests for other cases than (2,3)
        """
        return (
            self.is_sparse(K, L)
            and self.number_of_edges() == K * self.number_of_nodes() - L
        )

    @doc_category("Waiting for implementation")
    def zero_extension(self, vertices: List[Vertex], dim: int = 2) -> None:
        """
        Notes
        -----
        Modifies self only when explicitly required.
        """
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def one_extension(self, vertices: List[Vertex], edge: Edge, dim: int = 2) -> None:
        """
        Notes
        -----
        Modifies self only when explicitly required.
        """
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def k_extension(
        self, k: int, vertices: List[Vertex], edges: Edge, dim: int = 2
    ) -> None:
        """
        Notes
        -----
        Modifies self only when explicitly required.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def all_k_extensions(self, k: int, dim: int = 2) -> None:
        """
        Return list of all possible k-extensions of the graph.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def extension_sequence(self, dim: int = 2) -> Any:
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        raise NotImplementedError()

    @doc_category("Generic rigidity")
    def is_vertex_redundantly_rigid(self, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`vertex redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        return self.is_k_vertex_redundantly_rigid(1, dim)

    @doc_category("Generic rigidity")
    def is_k_vertex_redundantly_rigid(self, k: int, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`k-vertex redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`.

        TODO
        ----
        Avoid creating deepcopies by remembering the edges.
        Tests, examples.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if not isinstance(k, int):
            raise TypeError(f"k needs to be a nonnegative integer, but is {k}!")
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        for vertex_set in combinations(self.nodes, k):
            G = deepcopy(self)
            G.delete_vertices(vertex_set)
            if not G.is_rigid(dim):
                return False
        return True

    @doc_category("Generic rigidity")
    def is_redundantly_rigid(self, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`.
        """
        return self.is_k_redundantly_rigid(1, dim)

    @doc_category("Generic rigidity")
    def is_k_redundantly_rigid(self, k: int, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`k-redundantly (generically) dim-rigid
        <def-redundantly-rigid-graph>`.

        TODO
        ----
        Tests, examples.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if not isinstance(k, int):
            raise TypeError(f"k needs to be a nonnegative integer, but is {k}!")
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        for edge_set in combinations(self.edge_list(), k):
            self.delete_edges(edge_set)
            if not self.is_rigid(dim):
                self.add_edges(edge_set)
                return False
            self.add_edges(edge_set)
        return True

    @doc_category("Generic rigidity")
    def is_rigid(self, dim: int = 2, combinatorial: bool = True) -> bool:
        """
        Check whether the graph is :prf:ref:`(generically) dim-rigid <def-gen-rigid>`.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
        >>> G.is_rigid()
        False
        >>> G.add_edge(0,2)
        >>> G.is_rigid()
        True

        TODO
        ----
        Pebble game algorithm for d=2.

        Notes
        -----
         * dim=1: Connectivity
         * dim=2: Pebble-game/(2,3)-rigidity
         * dim>=1: Rigidity Matrix if ``combinatorial==False``
        By default, the graph is in dimension two and a combinatorial check is employed.
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if not isinstance(combinatorial, bool):
            raise TypeError(
                "combinatorial determines the method of rigidity-computation. "
                "It needs to be a Boolean."
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        elif dim == 1:
            return self.is_connected()
        elif dim == 2 and combinatorial:
            deficiency = -(2 * self.number_of_nodes() - 3) + self.number_of_edges()
            if deficiency < 0:
                return False
            else:
                for edge_subset in combinations(self.edges, deficiency):
                    H = self.edge_subgraph(
                        [edge for edge in self.edges if edge not in edge_subset]
                    )
                    if H.is_tight(2, 3):
                        return True
                return False
        elif not combinatorial:
            from pyrigi.framework import Framework

            F = Framework.Random(self, dim)
            return F.is_inf_rigid()
        else:
            raise ValueError(
                f"The Dimension for combinatorial computation must be either 1 or 2, "
                f"but is {dim}"
            )

    @doc_category("Generic rigidity")
    def is_min_rigid(self, dim: int = 2, combinatorial: bool = True) -> bool:
        """
        Check whether the graph is :prf:ref:`minimally (generically) dim-rigid
        <def-min-rigid-graph>`.

        By default, the graph is in dimension 2 and a combinatorial algorithm is applied.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0), (1,3)])
        >>> G.is_min_rigid()
        True
        >>> G.add_edge(0,2)
        >>> G.is_min_rigid()
        False

        Notes
        -----
         * dim=1: Tree
         * dim=2: Pebble-game/(2,3)-tight
         * dim>=1: Probabilistic Rigidity Matrix (maybe symbolic?)
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if not isinstance(combinatorial, bool):
            raise TypeError(
                "combinatorial determines the method of rigidity-computation. "
                "It needs to be a Boolean."
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        elif dim == 1:
            return self.is_tree()
        elif dim == 2 and combinatorial:
            return self.is_tight(2, 3)
        elif not combinatorial:
            from pyrigi.framework import Framework

            F = Framework.Random(self, dim)
            return F.is_min_inf_rigid()
        else:
            raise ValueError(
                f"The dimension for combinatorial computation must be either 1 or 2, "
                f"but is {dim}"
            )

    @doc_category("Generic rigidity")
    def is_globally_rigid(self, dim: int = 2) -> bool:
        """
        Check whether the graph is :prf:ref:`globally dim-rigid
        <def-globally-rigid-graph>`.

        TODO
        ----
        missing definition, implementation for dim>=3

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,0)])
        >>> G.is_globally_rigid()
        True

        Notes
        -----
         * dim=1: 2-connectivity
         * dim=2: redundantly rigid+3-connected
         * dim>=3: Randomized Rigidity Matrix => Stress (symbolic maybe?)
        By default, the graph is in dimension 2.
        A complete graph is automatically globally rigid
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        elif dim == 1:
            if (self.number_of_nodes() == 2 and self.number_of_edges() == 1) or (
                self.number_of_nodes() == 1 or self.number_of_nodes() == 0
            ):
                return True
            return self.vertex_connectivity() >= 2
        elif dim == 2:
            if (
                (self.number_of_nodes() == 3 and self.number_of_edges() == 3)
                or (self.number_of_nodes() == 2 and self.number_of_edges() == 1)
                or (self.number_of_nodes() == 1 or self.number_of_nodes() == 0)
            ):
                return True
            return self.is_redundantly_rigid() and self.vertex_connectivity() >= 3
        else:

            # Random sampling from [1,N] for N depending quadratically on number
            # of vertices.
            raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def is_Rd_dependent(self, dim: int = 2) -> bool:
        """
        Notes
        -----
         * dim=1: Graphic Matroid
         * dim=2: not (2,3)-sparse
         * dim>=1: Compute the rank of the rigidity matrix and compare with edge count
        """
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def is_Rd_independent(self, dim: int = 2) -> bool:
        """
        Notes
        -----
         * dim=1: Graphic Matroid
         * dim=2: (2,3)-sparse
         * dim>=1: Compute the rank of the rigidity matrix and compare with edge count
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def is_Rd_circuit(self, dim: int = 2) -> bool:
        """
        Notes
        -----
         * dim=1: Graphic Matroid
         * dim=2: Remove any edge and it becomes sparse
           (sparsity for every subgraph except whole graph?)
         * dim>=1: Dependent + Remove every edge and compute the rigidity matrix' rank
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        raise NotImplementedError()

    @doc_category("Waiting for implementation")
    def is_Rd_closed(self, dim: int = 2) -> bool:
        """
        Notes
        -----
         * dim=1: Graphic Matroid
         * dim=2: ??
         * dim>=1: Adding any edge does not increase the rigidity matrix rank
        """
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()
        raise NotImplementedError()

    @doc_category("Generic rigidity")
    def max_rigid_subgraphs(self, dim: int = 2) -> List[GraphType]:
        """
        List vertex-maximal rigid subgraphs of the graph.

        Definitions
        -----
        :prf:ref:`Maximal rigid subgraph <def-maximal-rigid-subgraph>`

        TODO
        ----
        missing definition

        Notes
        -----
        We only return nontrivial subgraphs, meaning that there need to be at
        least ``dim+1`` vertices present. If the graph itself is rigid, it is clearly
        maximal and is returned.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,0)])
        >>> G.max_rigid_subgraphs()
        []

        >>> G = Graph([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,2), (5,3)])
        >>> G.is_rigid()
        False
        >>> G.max_rigid_subgraphs()
        [Graph with vertices [0, 1, 2] and edges [[0, 1], [0, 2], [1, 2]], Graph with vertices [3, 4, 5] and edges [[3, 4], [3, 5], [4, 5]]]
        """  # noqa: E501
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        if self.number_of_nodes() <= dim:
            return []
        if self.is_rigid():
            return [self]
        max_subgraphs = []
        for vertex_subset in combinations(self.nodes, self.number_of_nodes() - 1):
            G = self.subgraph(vertex_subset)
            max_subgraphs = [
                j for i in [max_subgraphs, G.max_rigid_subgraphs(dim)] for j in i
            ]

        # We now remove the graphs that were found at least twice.
        clean_list = []
        for i in range(len(max_subgraphs)):
            iso_bool = False
            for j in range(i + 1, len(max_subgraphs)):
                if set(max_subgraphs[i].nodes) == set(
                    max_subgraphs[j].nodes
                ) and max_subgraphs[i].is_isomorphic(max_subgraphs[j]):
                    iso_bool = True
                    break
            if not iso_bool:
                clean_list.append(max_subgraphs[i])
        return clean_list

    @doc_category("Generic rigidity")
    def min_rigid_subgraphs(self, dim: int = 2) -> List[GraphType]:
        """
        List vertex-minimal non-trivial rigid subgraphs of the graph.

        Definitions
        -----
        :prf:ref:`Minimal rigid subgraph <def-minimal-rigid-subgraph>`

        TODO
        ----
        missing definition

        Notes
        -----
        We only return nontrivial subgraphs, meaning that there need to be at
        least ``dim+1`` vertices present.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (2,3), (3,4), (4,5), (5,0), (0,3), (4,1), (5,2)])
        >>> G.is_rigid()
        True
        >>> G.min_rigid_subgraphs()
        [Graph with vertices [0, 1, 2, 3, 4, 5] and edges [[0, 1], [0, 3], [0, 5], [1, 2], [1, 4], [2, 3], [2, 5], [3, 4], [4, 5]]]
        """  # noqa: E501
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(
                f"The dimension needs to be a positive integer, but is {dim}!"
            )
        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        min_subgraphs = []
        if self.number_of_nodes() <= 2:
            return []
        elif self.number_of_nodes() == dim + 1 and self.is_rigid():
            return [self]
        elif self.number_of_nodes() == dim + 1:
            return []
        for vertex_subset in combinations(self.nodes, self.number_of_nodes() - 1):
            G = self.subgraph(vertex_subset)
            subgraphs = G.min_rigid_subgraphs(dim)
            if len(subgraphs) == 0 and G.is_rigid():
                min_subgraphs.append(G)
            else:
                min_subgraphs = [
                    j for i in [min_subgraphs, G.min_rigid_subgraphs(dim)] for j in i
                ]

        # We now remove the graphs that were found at least twice.
        clean_list = []
        for i in range(len(min_subgraphs)):
            iso_bool = False
            for j in range(i + 1, len(min_subgraphs)):
                if set(min_subgraphs[i].nodes) == set(
                    min_subgraphs[j].nodes
                ) and min_subgraphs[i].is_isomorphic(min_subgraphs[j]):
                    iso_bool = True
                    break
            if not iso_bool:
                clean_list.append(min_subgraphs[i])
        # If no smaller graph is found and the graph is rigid, it is returned.
        if not clean_list and self.is_rigid():
            clean_list = [self]
        return clean_list

    @staticmethod
    def _find_triangle_components(
        graph: Graph,
    ) -> Tuple[Dict[Edge, int], List[List[Edge]]]:
        """
        Finds all the components of triangle equivalence.

        Returns mapping from edges to their component id (int)
        and the other way around from component to set of edges
        Order of vertices in the edge pairs is arbitrary.
        Components are indexed from 0
        """
        triangle_components = UnionFind[Edge]()
        for edge in graph.edges:
            v, u = edge

            # We cannot sort vertices and we cannot expect
            # any regularity or order of the vertices
            triangle_components.join((u, v), (v, u))

            vset = set([w for e in graph.edges(v) for w in e])
            uset = set([w for e in graph.edges(u) for w in e])
            intersection = vset.intersection(uset) - set([v, u])
            for w in intersection:
                triangle_components.join((u, v), (w, v))
                triangle_components.join((u, v), (w, u))

        edge_to_component: Dict[Edge, int] = {}
        component_to_edge: List[List[Edge]] = []

        for edge in graph.edges:
            root = triangle_components.find(edge)

            if root not in edge_to_component:
                edge_to_component[root] = id = len(component_to_edge)
                component_to_edge.append([])
            else:
                id = edge_to_component[root]

            edge_to_component[edge] = id
            component_to_edge[id].append(edge)

        return edge_to_component, component_to_edge

    @staticmethod
    def _create_line_graph_from_components(
        graph: Graph,
        edges_to_components: Dict[Edge, int],
    ) -> nx.Graph:
        """
        Creates a line graph from the components given.
        Each edge must belong to a component.
        Ids of these components are then used
        as vertexes of the new graph.
        Id's start from 0
        """

        # graph used to find NAC coloring easily
        line_graph = nx.Graph()

        def get_edge_component(e: Edge) -> int:
            u, v = e
            res = edges_to_components.get((u, v))
            if res is None:
                res = edges_to_components[(v, u)]
            return res

        for v in graph.vertex_list():
            edges = list(graph.edges(v))
            for i in range(0, len(edges)):
                c1 = get_edge_component(edges[i])

                for j in range(i + 1, len(edges)):
                    c2 = get_edge_component(edges[j])
                    if c1 == c2:
                        continue
                    elif c1 < c2:
                        line_graph.add_edge(c1, c2)
                    else:
                        line_graph.add_edge(c2, c1)

        return line_graph

    @staticmethod
    def _find_nac_coloring_naive(
        graph: Graph,
        line_graph: nx.Graph,
        component_to_edges: List[List[Edge]],
        limit: int | None,
    ) -> Tuple[List[NACColoring], int]:
        # the NAC coloring found
        coloringList: List[NACColoring] = []
        # number of the is_nac_coloring calls
        checks_cnt = 0

        vertices = list(line_graph.nodes())

        # iterate all the coloring variants
        # division by 2 is used as the problem is symmetrical
        for mask in range(1, 2 ** len(vertices) // 2):
            coloring: Tuple[Set[Edge], Set[Edge]] = (set(), set())
            for i, e in enumerate(vertices):
                (coloring[0] if mask & (1 << i) else coloring[1]).update(
                    component_to_edges[e]
                )

            checks_cnt += 1
            if not graph.is_nac_coloring(coloring):
                continue

            coloringList.append(coloring)

            # short circuit if the limit is reached
            if limit is not None and len(coloringList) >= limit:
                return (coloringList, checks_cnt)

        return (coloringList, checks_cnt)

    @staticmethod
    def _find_cycles(graph: nx.Graph, all: bool = False) -> Set[Tuple[int, ...]]:
        """
        For each vertex finds one/all of the shortest cycles it lays on

        Parameters
        ----------
        graph:
            The graph to work with, vertices should be integers indexed from 0
        all:
            if set to True, all the shortest cycles are returned
            Notice that for dense graphs the number of cycles can be quite huge
            if set to False, some cycle is returned for each vertex
            Defaults to False
        ----------
        """
        found_cycles: Set[Tuple[int, ...]] = set()

        vertices = list(graph.nodes)

        def insert_found_cycle(cycle: List[int]) -> None:
            """
            Runs post-processing on a cycle
            Makes sure the first element is the smallest one
            and the second one is greater than the last one.
            This is required for equality checks in set later.
            """
            # TODO use np.argmin

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
            queue = deque([start])
            parents = [-1 for _ in range(len(vertices))]
            parents[start] = start
            local_cycle_len = -1

            def backtrack(v: int, u: int) -> List[int]:
                """
                Reconstructs the found cycle
                """
                cycles: List[int] = []

                # reconstructs one part of the cycle
                cycles.append(u)
                p = parents[u]
                while p != start:
                    cycles.append(p)
                    p = parents[p]
                cycles = list(reversed(cycles))

                # and the other one
                cycles.append(v)
                p = parents[v]
                while p != start:
                    cycles.append(p)
                    p = parents[p]

                cycles.append(start)
                return cycles

            # typical BFS
            while queue:
                v = queue.popleft()
                parent = parents[v]

                # TODO consider shuffling
                for u in graph.neighbors(v):
                    # so I don't create cycle on 1 edge
                    # this could be done sooner, I know...
                    if u == parent:
                        continue

                    # newly found item
                    if parents[u] == -1:
                        parents[u] = v
                        queue.append(u)
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
            bfs(start)

        return found_cycles

    @staticmethod
    def _find_nac_coloring_cycles(
        graph: Graph,
        line_graph: nx.Graph,
        component_to_edges: List[List[Edge]],
        limit: int | None,
    ) -> Tuple[List[NACColoring], int]:
        # the NAC coloring found
        coloringList: List[NACColoring] = []
        # number of the is_nac_coloring calls
        checks_cnt = 0

        vertices = list(line_graph.nodes())
        # so we start with 0
        vertices.sort()

        # find some small cycles for state filtering
        cycles = Graph._find_cycles(line_graph, all=False)
        # the idea is that smaller cycles reduce the state space more
        cycles = sorted(cycles, key=lambda c: len(c))

        def create_bitmask(cycle: Tuple[int, ...]) -> int | None:
            mask = 0
            for v in cycle:
                # # This cycle contains a triangle component,
                # # this trick cannot be used
                # if len(component_to_edges[v]) > 1:
                #     return None

                mask |= 1 << v
            print(cycle, bin(mask))
            return mask

        cycle_masks = list(filter(None, [create_bitmask(c) for c in cycles]))

        # holds all the vertices that represent more edges
        triangle_components_mask = 0
        for v in vertices:
            if len(component_to_edges[v]) > 1:
                triangle_components_mask |= 1 << v

        # iterate all the coloring variants
        # division by 2 is used as the problem is symmetrical
        for mask in range(1, 2 ** len(vertices) // 2):

            # Cycles checking
            # in this section we check trivial cycles if they are correct
            # before checking the whole graph
            wrong_mask_found = False

            for template in cycle_masks:
                stamp1, stamp2 = mask & template, (~mask) & template
                cnt1, cnt2 = stamp1.bit_count(), stamp2.bit_count()
                stamp, cnt = (stamp1, cnt1) if cnt1 == 1 else (stamp2, cnt2)

                if cnt != 1:
                    continue

                # now we know there is one node that has a wrong color
                # we check if the node is a triangle component
                # if so, we cannot skip this run
                if (stamp & triangle_components_mask) > 0:
                    continue

                wrong_mask_found = True
                break

            if wrong_mask_found:
                continue

            coloring: Tuple[Set[Edge], Set[Edge]] = (set(), set())
            for i, e in enumerate(vertices):
                (coloring[0] if mask & (1 << i) else coloring[1]).update(
                    component_to_edges[e]
                )

            checks_cnt += 1
            if not graph.is_nac_coloring(coloring):
                continue

            coloringList.append(coloring)

            # short circuit if the limit is reached
            if limit is not None and len(coloringList) >= limit:
                return (coloringList, checks_cnt)

        return (coloringList, checks_cnt)

    @doc_category("Generic rigidity")
    def find_nac_coloring(
        self, limit: int | None = 1, algorithm: str = "cycles"
    ) -> List[NACColoring]:
        """
        Finds a NAC-coloring of this graph if there exists one.
        Returns a list of NAC colorings found (certificates)
        up to the limit given in an unspecified order.

        Parameters
        ----------
        limit:
            Maximum number of colorings to search for.
            Use `None` for unlimited search.
            The value should be positive.
        algorithm:
            some options may provide better performance
            - naive - basic implementation, previous SOA
            - cycles - finds some small cycles and uses them to reduce state space
        ----------

        TODO example
        """
        assert limit is None or limit >= 1
        if algorithm not in ["naive", "cycles"]:
            raise ValueError(f"Unknown algorighm type: {algorithm}")

        if nx.number_of_selfloops(self) > 0:
            raise LoopError()

        if self.vertex_list() == 0:
            # TODO make own error type later
            raise "Undefined for an empty graph"
        # TODO graph with 1 vertex

        if nx.is_directed(self):
            raise "Cannot process a directed graph"

        # TODO to implement
        # find cycles of 4/5 and use pseudo CSP
        # coloring caching

        # TODO return this instead of a boolean flag
        class NACReason(Enum):
            """Result of a NAC coloring search"""

            NONE_EXISTS = False
            # NOT_CONNECTED = True
            NOT_2_VERTEX_CONNECTED = True
            # NOT_3_EDGE_CONNECTED = True
            CASE_FOUND = True

        # I'm not sure how to generate all the certificates,
        # so if more are requested, I fallback to the main solver.
        if limit == 1 and not nx.algorithms.connectivity.node_connectivity(self) >= 2:
            print("NOT_2_VERTEX_CONNECTED")
            generator = nx.algorithms.biconnected_components(self)
            component: Set[Vertex] = next(generator)
            assert next(generator)  # make sure there are more components

            red, blue = set(), set()
            for v, u in self.edges:
                (red if v in component and u in component else blue).add((u, v))

            return [(red, blue)]

        # TODO consult with Legersky
        # if not nx.algorithms.connectivity.is_k_edge_connected(self, 3):
        #     print("NOT_3_EDGE_CONNECTED")
        #     return (NACReason.NOT_3_EDGE_CONNECTED.value, None)

        edge_to_component, component_to_edge = Graph._find_triangle_components(self)
        line_graph = Graph._create_line_graph_from_components(self, edge_to_component)

        match algorithm:
            case "naive":
                res = Graph._find_nac_coloring_naive(
                    self, line_graph, component_to_edge, limit
                )
            case "cycles":
                res = Graph._find_nac_coloring_cycles(
                    self, line_graph, component_to_edge, limit
                )
            case _:
                raise ValueError(f"Unknown algorighm type: {algorithm}")
        # print("Process took", res[1], "iterations")
        return res[0]

    @doc_category("Generic rigidity")
    def is_nac_coloring(self, colors: NACColoring) -> bool:
        """
        Check if the coloring given is a NAC coloring.
        The algorithm checks if all the edges are in the same component.
        (TODO format)
        """

        # Both colors have to be used
        if len(colors[0]) == 0 or len(colors[1]) == 0:
            return False

        # We should rather check if the edges match exactly,
        # but that would be a little slower
        if len(colors[0]) + len(colors[1]) != len(self.edges):
            return False
        if len(colors[0].intersection(colors[1])) != 0:
            return False

        G = Graph()
        G.add_vertices(self.vertex_list())

        def check_coloring(red: Set[Edge], blue: Set[Edge]) -> bool:
            G.clear_edges()
            G.add_edges(red)

            component_mapping: Dict[Vertex, int] = {}
            vertices: Set[Vertex]
            for i, vertices in enumerate(nx.components.connected_components(G)):
                for v in vertices:
                    component_mapping[v] = i

            for e1, e2 in blue:
                if component_mapping[e1] == component_mapping[e2]:
                    return False
            return True

        return check_coloring(colors[0], colors[1]) and check_coloring(
            colors[1], colors[0]
        )

    @doc_category("General graph theoretical properties")
    def is_isomorphic(self, graph: GraphType) -> bool:
        """
        Check whether two graphs are isomorphic.

        Notes
        -----
        For further details, see :func:`networkx.algorithms.isomorphism.is_isomorphic`.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G_ = Graph([('b','c'), ('c','a')])
        >>> G.is_isomorphic(G_)
        True
        """
        return nx.is_isomorphic(self, graph)

    @doc_category("Other")
    def to_int(self, vertex_order: List[Vertex] = None) -> int:
        r"""
        Return the integer representation of the graph.

        The graph integer representation is the integer whose binary
        expansion is given by the sequence obtained by concatenation
        of the rows of the upper triangle of the adjacency matrix,
        excluding the diagonal.

        Parameters
        ----------
        vertex_order:
            By listing vertices in the preferred order, the adjacency matrix
            is computed with the given order. If no vertex order is
            provided, :meth:`~.Graph.vertex_list()` is used.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2)])
        >>> G.adjacency_matrix()
        Matrix([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]])
        >>> G.to_int()
        5

        TODO
        ----
        Implement taking canonical before computing the integer representation.
        Tests.
        """
        if self.number_of_edges() == 0:
            raise ValueError(
                "The integer representation is only defined "
                "for graphs with at least one edge."
            )
        if self.min_degree() == 0:
            raise ValueError(
                "The integer representation only works "
                "for graphs without isolated vertices."
            )
        if nx.number_of_selfloops(self) == 0:
            M = self.adjacency_matrix(vertex_order)
            upper_diag = [
                str(b) for i, row in enumerate(M.tolist()) for b in row[i + 1 :]
            ]
            return int("".join(upper_diag), 2)
        else:
            raise LoopError()

    @classmethod
    @doc_category("Class methods")
    def from_int(cls, N: int) -> GraphType:
        """
        Return a graph given its integer representation.

        See :meth:`to_int` for the description
        of the integer representation.
        """
        if not isinstance(N, int):
            raise TypeError(f"The parameter n has to be an integer, not {type(N)}.")
        if N <= 0:
            raise ValueError(f"The parameter n has to be positive, not {N}.")
        L = bin(N)[2:]
        n = math.ceil((1 + math.sqrt(1 + 8 * len(L))) / 2)
        rows = []
        s = 0
        L = "".join(["0" for _ in range(int(n * (n - 1) / 2) - len(L))]) + L
        for i in range(n):
            rows.append(
                [0 for _ in range(i + 1)] + [int(k) for k in L[s : s + (n - i - 1)]]
            )
            s += n - i - 1
        adjMatrix = Matrix(rows)
        return Graph.from_adjacency_matrix(adjMatrix + adjMatrix.transpose())

    @classmethod
    @doc_category("Class methods")
    def from_adjacency_matrix(cls, M: Matrix) -> GraphType:
        """
        Create a graph from a given adjacency matrix.

        Examples
        --------
        >>> M = Matrix([[0,1],[1,0]])
        >>> G = Graph.from_adjacency_matrix(M)
        >>> print(G)
        Graph with vertices [0, 1] and edges [[0, 1]]
        """
        if not M.is_square:
            raise TypeError("The matrix is not square!")
        if not M.is_symmetric():
            raise TypeError("The matrix is not symmetric.")

        vertices = range(M.cols)
        edges = []
        for i, j in combinations(vertices, 2):
            if not (M[i, j] == 0 or M[i, j] == 1):
                raise TypeError(
                    "The provided adjacency matrix contains entries other than 0 and 1"
                )
            if M[i, j] == 1:
                edges += [(i, j)]
        return Graph.from_vertices_and_edges(vertices, edges)

    @doc_category("General graph theoretical properties")
    def adjacency_matrix(self, vertex_order: List[Vertex] = None) -> Matrix:
        """
        Return the adjacency matrix of the graph.

        Parameters
        ----------
        vertex_order:
            By listing vertices in the preferred order, the adjacency matrix
            can be computed in a way the user expects. If no vertex order is
            provided, :meth:`~.Graph.vertex_list()` is used.

        Examples
        --------
        >>> G = Graph([(0,1), (1,2), (1,3)])
        >>> G.adjacency_matrix()
        Matrix([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0]])

        Notes
        -----
        :func:`networkx.linalg.graphmatrix.adjacency_matrix`
        requires `scipy`. To avoid unnecessary imports, the method is implemented here.
        """
        if vertex_order is None:
            vertex_order = self.vertex_list()
        else:
            if not set(self.nodes) == set(
                vertex_order
            ) or not self.number_of_nodes() == len(vertex_order):
                raise IndexError(
                    "The vertex_order must contain the same vertices as the graph!"
                )

        row_list = [
            [+((v1, v2) in self.edges) for v2 in vertex_order] for v1 in vertex_order
        ]

        return Matrix(row_list)

    @doc_category("Other")
    def random_framework(
        self, dim: int = 2, rand_range: Union(int, List[int]) = None
    ) -> FrameworkType:
        """
        Return framework with random realization.

        This method calls :meth:`.Framework.Random`.
        """
        from pyrigi.framework import Framework

        return Framework.Random(self, dim, rand_range)


Graph.__doc__ = Graph.__doc__.replace(
    "METHODS",
    generate_category_tables(
        Graph,
        1,
        [
            "Attribute getters",
            "Class methods",
            "Graph manipulation",
            "General graph theoretical properties",
            "Generic rigidity",
            "Sparseness",
            "Other",
            "Waiting for implementation",
        ],
        include_all=False,
    ),
)
