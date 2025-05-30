from collections import defaultdict
from itertools import product
import logging
import random
import networkx as nx
from pyrigi.data_type import Vertex
import pytest
import numpy as np

from pyrigi import Graph as PRGraph
from stablecut.flexible_graphs import stable_cut_in_flexible_graph
from stablecut.types import StableCut
from stablecut.util import (
    is_stable_cutset,
    is_stable_cutset_dividing,
    is_separating_set,
    is_separating_set_dividing,
    stable_set_violation,
    _revertable_set_removal,
)


@pytest.mark.stable_cutset
def test_stable_set_eq():
    set1 = StableCut({1, 2}, {3, 4}, {5})
    set2 = StableCut({3, 4}, {1, 2}, {5})
    set3 = StableCut({3, 4}, {1, 2}, {6})
    assert set1 == set1
    assert set1 == set2
    assert set1 != set3


def test__revertable_set_removal():
    graph1 = nx.Graph([(0, 1), (1, 2), (2, 3)])
    graph2 = graph1.copy()
    noop = lambda _: None
    _revertable_set_removal(graph2, set(), noop)
    assert nx.is_isomorphic(graph1, graph2)
    _revertable_set_removal(graph2, {2}, noop)
    assert nx.is_isomorphic(graph1, graph2)
    _revertable_set_removal(graph2, {0, 1, 2, 3}, noop)
    assert nx.is_isomorphic(graph1, graph2)
    _revertable_set_removal(nx.induced_subgraph(graph2, [0, 1, 2]), {1}, noop)
    assert nx.is_isomorphic(graph1, graph2)


@pytest.mark.stable_cutset
def test_stable_set_violation():
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])

    assert stable_set_violation(graph, {0, 1, 2}) in [(0, 1), (1, 2)]
    assert stable_set_violation(graph, {0, 1}) == (0, 1)
    assert stable_set_violation(graph, {0, 2}) is None


@pytest.mark.stable_cutset
def test_is_separating_set():
    graph = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (0, 4), (1, 4)])

    assert is_separating_set(graph, {2})
    assert is_separating_set(graph, {0, 2})
    assert is_separating_set(graph, {0, 1})
    assert not is_separating_set(graph, set())
    assert not is_separating_set(graph, {4, 3})

    assert is_separating_set_dividing(graph, {2}, 0, 3)
    assert is_separating_set_dividing(graph, {2}, 4, 3)
    assert not is_separating_set_dividing(graph, {2}, 0, 1)

    with pytest.raises(ValueError):
        is_separating_set_dividing(graph, {2}, 0, 2)


@pytest.mark.stable_cutset
def test_stable_cut_in_flexible_graph_edge_cases():
    # empty graph
    graph = nx.Graph()
    cut = stable_cut_in_flexible_graph(graph)
    assert cut is None

    # more vertices
    graph = PRGraph.from_vertices_and_edges([0, 1, 2], [])
    cut = stable_cut_in_flexible_graph(graph)
    assert cut is not None
    assert is_stable_cutset(graph, cut)

    # single vertex graph
    graph = PRGraph.from_vertices_and_edges([0], [])
    cut = stable_cut_in_flexible_graph(graph)
    assert cut is None

    # single edge graph
    graph = PRGraph.from_vertices_and_edges([0, 1], [(0, 1)])
    cut = stable_cut_in_flexible_graph(graph)
    assert cut is None

    # triangle graph
    graph = PRGraph.from_vertices_and_edges([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    cut = stable_cut_in_flexible_graph(graph)
    assert cut is None


@pytest.mark.stable_cutset
def test_stable_cut_in_flexible_graph():
    from pyrigi import Graph as Graph

    graph = PRGraph.from_vertices_and_edges(
        list(range(8)),
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (3, 7),
        ],
    )
    # print(nx.nx_agraph.to_agraph(graph))
    assert not graph.is_rigid()

    cut = stable_cut_in_flexible_graph(graph)
    assert cut is not None
    assert is_stable_cutset(graph, cut)

    cut = stable_cut_in_flexible_graph(graph, 0)
    assert cut is not None
    assert is_stable_cutset(graph, cut)

    cut = stable_cut_in_flexible_graph(graph, 0, 1)
    assert cut is None

    cut = stable_cut_in_flexible_graph(graph, 0, 4)
    assert cut is None

    for i in [5, 6, 7]:
        cut = stable_cut_in_flexible_graph(graph, 0, i)
        assert cut is not None
        assert is_stable_cutset(graph, cut)
    for i in [0, 1, 2]:
        cut = stable_cut_in_flexible_graph(graph, 7, i)
        assert cut is not None
        assert is_stable_cutset(graph, cut)


@pytest.mark.stable_cutset
def test_stable_cut_in_flexible_graph_prism():
    from pyrigi.graphDB import ThreePrism

    graph = ThreePrism()

    for u, v in product(graph.nodes, graph.nodes):
        cut = stable_cut_in_flexible_graph(graph, u, v)
        assert cut is None


@pytest.mark.stable_cutset
@pytest.mark.slow_main
@pytest.mark.parametrize(
    ("n", "p"), [(4, 0.3), (8, 0.3), (13, 0.4), (16, 0.4), (16, 0.4)]
)
@pytest.mark.parametrize("graph_no", [69])
@pytest.mark.parametrize("seed", [42, None])
@pytest.mark.parametrize("connected", [True, False])
def test_fuzzy_stable_cut_in_flexible_graph(
    n: int,
    p: float,
    graph_no: int,
    seed: int | None,
    connected: bool,
):
    from pyrigi import Graph

    pairs_per_graph = int(np.sqrt(n))

    rand = random.Random(seed)
    graphs = 0
    while graphs < graph_no:
        graph = Graph(nx.gnp_random_graph(n, p, seed=rand.randint(0, 2**30)))

        # Filter out unreasonable graphs
        if nx.is_connected(graph) != connected or graph.is_rigid():
            continue

        # Create mapping from vertex to rigid component ids
        # used later to assert expected result
        rigid_components = graph.rigid_components()
        vertex_to_comp_id: dict[Vertex, set[int]] = defaultdict(set)
        for i, comp in enumerate(rigid_components):
            for v in comp:
                vertex_to_comp_id[v].add(i)

        # Makes sure that the underlying code does not require pyrigi.Graph
        graph = nx.Graph(graph)

        # Take first pairs_per_graph pairs of vertices
        pairs = list(product(graph.nodes, graph.nodes))
        rand.shuffle(pairs)

        tests_negative = 0
        tests_positive = 0
        for pair in pairs:
            u, v = pair

            # makes sure both types of tests are run enough
            if (
                tests_positive > pairs_per_graph * 2 // 3
                and tests_negative > pairs_per_graph * 1 // 3
            ):
                break

            try:
                if u == v or (vertex_to_comp_id[u] & vertex_to_comp_id[v]):
                    # invalid input
                    logging.disable(logging.WARNING)
                    cut = stable_cut_in_flexible_graph(graph, u, v)
                    logging.disable(0)

                    assert cut is None
                    tests_negative += 1
                else:
                    # valid input
                    cut = stable_cut_in_flexible_graph(graph, u, v)
                    assert cut is not None
                    assert is_stable_cutset_dividing(graph, cut, u, v)
                    tests_positive += 1

            except AssertionError as e:
                # Changes the error message to include graph
                error_message = f"Assertion failed: {e}\nGraph: {nx.nx_agraph.to_agraph(graph)}\nComponents: {rigid_components}\nVertices: [{u} {v}]"
                raise AssertionError(error_message) from None

        graphs += 1
