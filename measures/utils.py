import numpy as np


def _average_statistic(statistic, sentence_graphs):
    """Calculate the statistic for every sentence and return mean and
    standard deviation.
    """
    results = [statistic(g) for g in sentence_graphs]
    return np.mean(results), np.std(results)
