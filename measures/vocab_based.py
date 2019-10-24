"""
In the last sixty years there have been a series of calculation proposals for measuring the lexical richness of a text. This richness gives us an idea of the number of different terms used in a text and the diversity of the vocabulary.

The above text is from "Lexical Statistics and Tipological Structures: A Measure of Lexical Richness" by Joan Torruella and Ramon Capsada.
"""

from typing import List
import collections
import numpy as np

## First class of indices based on the direct relationship between the number of terms and words (type-token).


def type_token_ratio(txt_len: int, vocab_size: int) -> float:
    """
    TTR (type-token ratio), by Templin, 1957.
    """
    return vocab_size / txt_len


def guiraud_r(txt_len: int, vocab_size: int) -> np.float64:
    """
    The TTR formula underwent simple corrections: RTTR (root type-token ratio), Guiraud, 1960.
    """
    return vocab_size / np.sqrt(txt_len)


def cttr(txt_len: int, vocab_size: int) -> np.float64:
    """
    The TTR formula underwent simple corrections: CTTR (corrected type token ratio) by Carrol, 1964.
    """
    return vocab_size / np.sqrt(2 * txt_len)


## Second class of indices has been developed using formular based n logarithmic function. The functions below grows in such a way as to adapt better to the behaviour of the relation that exists between the ters (types_ and the total number of words in a text (tokens).


def herdan_c(txt_len: int, vocab_size: int) -> np.float64:
    """
    H index developed by Herdan, 1960.
    """
    return np.log(vocab_size) / np.log(txt_len)


def dugast_k(txt_len: int, vocab_size: int) -> np.float64:
    """
    K index developed by Dugast.
    """
    return np.log(vocab_size) / np.log(np.log(txt_len))


def maas_a2(txt_len, vocab_size) -> np.float64:
    """
    M index developed by Mass, 1966. This displays most stability with respect to the text length.
    """
    return (np.log(txt_len) - np.log(vocab_size)) / (np.log(txt_len) ** 2)


def dugast_u(txt_len: int, vocab_size: int) -> np.float64:
    """
    U Developed by Dugast, 1978.
    """
    return (np.log(txt_len) ** 2) / (np.log(txt_len) - np.log(vocab_size))


def tuldava_ln(txt_len: int, vocab_size: int) -> np.float64:
    """
    T index developed by Tuldava, 1993.
    """
    return (1 - (vocab_size ** 2)) / ((vocab_size ** 2) * np.log(txt_len))


def summer_s(txt_len: int, vocab_size: int) -> np.float64:
    """
    S index developed by Summer, 1966.
    """
    return np.log(np.log(vocab_size)) / np.log(np.log(txt_len))


## Third class of indices is formed by a group of indices obtained from more complex calculations.
def sttr(tokens: List[str], window_size: int = 100) -> np.float64:
    """calculate standardized type-token ratio
    originally Kubat&Milicka 2013. Much better explained
    in Evert et al. 2017.
    :param ci:  additionally calculate and return the confidence interval, returns a tuple

    In this process the text to be analysed is divided into equal segments in terms of the number of words (normally 100 words per segment). For each segment the TTR is calculated and using an arithmetic mean of the TTR for each segment the MSTTR is obtained.
    """
    results = []
    for i in range(int(len(tokens) / window_size)):  # ignore last partial chunk
        txt_len, vocab_size = preprocess(
            tokens[i * window_size : (i * window_size) + window_size]
        )
        results.append(type_token_ratio(txt_len, vocab_size))
    return np.mean(results)
