import numpy as np
from utils import _average_statistic


"""
TODO: add type hinting
"""
def average_sentence_length(sentence_graphs):
    return _average_statistic(sentence_length, sentence_graphs)


def sentence_length(g):
    return len(g)


def average_sentence_length_characters(sentence_graphs):
    return _average_statistic(sentence_length_characters, sentence_graphs)


def sentence_length_characters(g):
    """Sum of token lengths plus number of token boundaries, i.e. we
    assume a space between all tokens.
    """
    tokens = [l["token"] for v, l in g.nodes(data=True)]
    token_lengths = vocabulary_richness.average_token_length_characters(tokens, raw=True)
    return sum(token_lengths) + len(token_lengths) - 1


def average_sentence_length_syllables(sentence_graphs, lang="en_EN"):
    tokens = [l["token"] for g in sentence_graphs for v, l in g.nodes(data=True)]
    token_lengths = vocabulary_richness.average_token_length_syllables(tokens, lang, raw=True)
    position = 0
    sentence_lengths = []
    for g in sentence_graphs:
        sentence_lengths.append(sum(token_lengths[position:position + len(g)]))
        position += len(g)
    return np.mean(sentence_lengths), np.std(sentence_lengths)
