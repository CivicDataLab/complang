import numpy as np
from utils import _average_statistic

# TODO: add type hinting

def average_average_dependency_distance(sentence_graphs):
    """Oya (2011)"""
    return _average_statistic(average_dependency_distance, sentence_graphs)


def average_dependency_distance(g):
    """Oya (2011)"""
    distances = dependency_distances(g)
    if len(distances) > 0:
        return statistics.mean(distances)
    else:
        return 0


def dependency_distances(g):
    """Return all dependency distances."""
    distances = []
    for s, t in g.edges(data=False):
        distances.append(abs(s - t))
    return distances
