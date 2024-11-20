from typing import *

import networkx as nx


class NiceGraph(nx.Graph):
    def __str__(self) -> str:
        return graph_str(self)

    def __repr__(self) -> str:
        return graph_str(self)


def graph_str(graph: nx.Graph) -> str:
    return f"Graph (|V|={graph.number_of_nodes()},|E|={graph.number_of_edges()}) ({list(graph.nodes)} {list(graph.edges)})"
