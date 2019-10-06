from utils import _average_statistic
import numpy as np


def average_punctuation_per_sentence(sentence_graphs):
    return _average_statistic(punctuation_per_sentence, sentence_graphs)


def punctuation_per_sentence(g):
    punctuation = set(["$.", "$,", "$("])
    return len([v for v, l in g.nodes(data=True) if l["pos"] in punctuation])
