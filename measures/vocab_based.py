from typing import List
import collections
import numpy as np


def type_token_ratio(txt_len: int, vocab_size: int) -> float:
    return vocab_size / txt_len

def guiraud_r(txt_len: int, vocab_size: int) -> np.float64:
    return vocab_size / np.sqrt(txt_len)

def herdan_c(txt_len: int, vocab_size: int) -> np.float64:
    return np.log(vocab_size) / np.log(txt_len)

def dugast_k(txt_len: int, vocab_size: int) -> np.float64:
    return np.log(vocab_size) / np.log(np.log(txt_len))

def maas_a2(txt_len, vocab_size) -> np.float64:
    return (np.log(txt_len) - np.log(vocab_size)) / (np.log(txt_len) ** 2)

def dugast_u(txt_len: int, vocab_size: int) -> np.float64:
    return (np.log(txt_len) ** 2) / (np.log(txt_len) - np.log(vocab_size))


def tuldava_ln(txt_len: int, vocab_size: int) -> np.float64:
    return (1 - (vocab_size ** 2)) / ((vocab_size ** 2) * np.log(txt_len))

# Check
def brunet_w(txt_len: int, vocab_size: int) -> float:
    """Brunet (1978)"""
    a = -0.172
    return txt_len ** (vocab_size ** -a)

def cttr(txt_len: int, vocab_size: int) -> np.float64:
    """Carroll's Corrected Type-Token Ration"""
    return vocab_size / np.sqrt(2 * txt_len)


def summer_s(txt_len: int, vocab_size: int) -> np.float64:
    """Summer's S index"""
    return np.log(np.log(vocab_size)) / np.log(np.log(txt_len))


def sttr(tokens: List[str], window_size: int = 1000, ci: bool=False) -> np.float64:
    """calculate standardized type-token ratio
    originally Kubat&Milicka 2013. Much better explained
    in Evert et al. 2017.
    :param ci:  additionally calculate and return the confidence interval, returns a tuple
    """
    results = []
    for i in range(int(len(tokens) / window_size)):  # ignore last partial chunk
        txt_len, vocab_size = preprocess(tokens[i * window_size:(i * window_size) + window_size])
        results.append(type_token_ratio(txt_len, vocab_size))
    if ci:
        return (np.mean(results), _sttr_ci(results))
    return np.mean(results)
