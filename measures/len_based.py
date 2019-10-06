from typing import List
import numpy as np
import pyphen


def average_token_length_characters(
    tokens: List[str], stdev: bool = True, raw: bool = False
) -> np.float64:
    """Average token length in characters."""
    token_lengths = [len(t) for t in tokens]
    if raw:
        return token_lengths
    mean_length = np.mean(token_lengths)
    if stdev:
        return mean_length, np.std(token_lengths)
    return mean_length


def average_token_length_syllables(
    tokens: List[str], lang="en_EN", stdev: bool = True, raw: bool = False
) -> np.float64:
    """Average token length in syllables. Pyphen uses the Hunspell
    hyphenation dictionaries
    """
    dic = pyphen.Pyphen(lang=lang)
    token_lengths = []
    for token in tokens:
        hyphens = dic.positions(token)
        token_lengths.append(len(hyphens) + 1)
    if raw:
        return token_lengths
    mean_length = np.mean(token_lengths)
    if stdev:
        return mean_length, np.std(token_lengths)
    return mean_length
