import numpy as np
from utils import _average_statistic

"""
TODO: add type hinting
"""
def average_punctuation_per_sentence(sentence_graphs):
    return _average_statistic(punctuation_per_sentence, sentence_graphs)


def punctuation_per_sentence(g):
    punctuation = set(["$.", "$,", "$("])
    return len([v for v, l in g.nodes(data=True) if l["pos"] in punctuation])


def average_punctuation_per_token(sentence_graphs):
    punct, tokens = 0, 0
    for g in sentence_graphs:
        tokens += len(g)
        punct += punctuation_per_sentence(g)
    return punct / tokens
