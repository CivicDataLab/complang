import numpy as np
from utils import _average_statistic
import networkx


def average_closeness_centrality(sentence_graphs):
    """Closeness centrality of the root vertex, i.e. the inverse of the
    average length of the shortest paths from the root to all other
    vertices. Used by Oya (2012).
    """
    return _average_statistic(closeness_centrality, sentence_graphs)


def closeness_centrality(g):
    """Closeness centrality of the root vertex, i.e. the inverse of the
    average length of the shortest paths from the root to all other
    vertices. Used by Oya (2012).
    """
    if len(g) > 1:
        root = [v for v, l in g.nodes(data=True) if "root" in l][0]
        return networkx.algorithms.centrality.closeness_centrality(g, root, reverse=True)
    else:
        return 1


def average_outdegree_centralization(sentence_graphs):
    """Outdegree centralization of the graph (Freeman, 1978). Return
    values range between 0 and 1. 1 means all other vertices are
    dependent on the root vertex. Used by Oya (2012).
    """
    return _average_statistic(outdegree_centralization, sentence_graphs)


def outdegree_centralization(g):
    """Outdegree centralization of the graph (Freeman, 1978). Return
    values range between 0 and 1. 1 means all other vertices are
    dependent on the root vertex. Used by Oya (2012).
    """
    if len(g) > 1:
        out_degrees = [deg for v, deg in g.out_degree()]
        max_out_degree = max(out_degrees)
        # for directed graphs, the denominator should be n² - 2n + 1
        # instead of n² - 3n + 2
        centr = sum(max_out_degree - deg for deg in out_degrees) / (len(g) ** 2 - 2 * len(g) + 1)
        assert centr <= 1
        return centr
    else:
        return 1


def average_closeness_centralization(sentence_graphs):
    """Closeness centralization of the graph (Freeman, 1978). Return
    values range between 0 and 1. 1 means all other vertices are
    dependent on the root vertex. Used by Oya (2012).
    """
    return _average_statistic(closeness_centralization, sentence_graphs)


def closeness_centralization(g):
    """Closeness centralization of the graph (Freeman, 1978). Return
    values range between 0 and 1. 1 means all other vertices are
    dependent on the root vertex. Used by Oya (2012).
    """
    if len(g) > 1:
        cc = networkx.algorithms.centrality.closeness_centrality(g, reverse=True).values()
        max_cc = max(cc)
        # for directed graphs, the denominator should be n - 1
        # instead of (n² - 3n + 2)/(2n - 3)
        centr = sum(max_cc - c for c in cc) / (len(g) - 1)
        assert centr <= 1
        return centr
    else:
        return 1


def average_longest_shortest_path(sentence_graphs):
    return _average_statistic(longest_shortest_path, sentence_graphs)


def longest_shortest_path(g):
    """Longest shortest path from the root vertex, i.e. depth of the
    tree.

    """
    if len(g) > 1:
        root = [v for v, l in g.nodes(data=True) if "root" in l][0]
        return max(networkx.algorithms.shortest_path_length(g, source=root).values())
    else:
        return 0
