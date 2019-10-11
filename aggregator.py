from typing import List
import collections
import numpy as np
from measures.freq_based import sichel_s
from measures.freq_based import michea_m
from measures.freq_based import honore_h
from measures.freq_based import entropy
from measures.freq_based import yule_k
from measures.freq_based import simpson_d
from measures.freq_based import herdan_vm
from measures.freq_based import hdd
from measures.probmodel_based import orlov_z
from measures.other_measures import mtld
from measures.vocab_based import type_token_ratio
from measures.vocab_based import guiraud_r
from measures.vocab_based import herdan_c
from measures.vocab_based import dugast_k
from measures.vocab_based import maas_a2
from measures.vocab_based import dugast_u
from measures.vocab_based import tuldava_ln
from measures.vocab_based import brunet_w
from measures.vocab_based import cttr
from measures.vocab_based import summer_s
from measures.helper import _sttr_ci


def preprocess(tokens, fs=False):
    """Return text length, vocabulary size and optionally the frequency
    spectrum.

    :param fs: additionally calculate and return the frequency
               spectrum

    """
    txt_len = len(tokens)
    vocab_size = len(set(tokens))
    if fs:
        frequency_list = collections.Counter(tokens)
        freq_spectrum = dict(collections.Counter(frequency_list.values()))
        return txt_len, vocab_size, freq_spectrum
    return txt_len, vocab_size


def bootstrap(
    tokens: List[str],
    measure: str = "type_token_ratio",
    window_size: int = 3,
    ci: bool = False,
    raw=False,
):
    """calculate bootstrap for lex diversity measures
    as explained in Evert et al. 2017. if measure='type_token_ratio'
    it calculates standardized type-token ratio
    :param ci:  additionally calculate and return the confidence interval
    returns a tuple
    :param raw:  return the raw results
    """
    results = []
    measures = dict(
        type_token_ratio=type_token_ratio,
        guiraud_r=guiraud_r,
        herdan_c=herdan_c,
        dugast_k=dugast_k,
        maas_a2=maas_a2,
        dugast_u=dugast_u,
        tuldava_ln=tuldava_ln,
        brunet_w=brunet_w,
        cttr=cttr,
        summer_s=summer_s,
        sichel_s=sichel_s,
        michea_m=michea_m,
        honore_h=honore_h,
        entropy=entropy,
        yule_k=yule_k,
        simpson_d=simpson_d,
        herdan_vm=herdan_vm,
        hdd=hdd,
        orlov_z=orlov_z,
        mtld=mtld,
    )
    # tl_vs: txt_len, vocab_size
    # vs_fs: vocab_size, freq_spectrum
    # tl_vs_fs: txt_len, vocab_size, freq_spectrum
    # tl_fs: txt_len, freq_spectrum
    # t: tokens
    classes = dict(
        tl_vs=(
            "type_token_ratio",
            "guiraud_r",
            "herdan_c",
            "dugast_k",
            "maas_a2",
            "dugast_u",
            "tuldava_ln",
            "brunet_w",
            "cttr",
            "summer_s",
        ),
        vs_fs=("sichel_s", "michea_m"),
        tl_vs_fs=("honore_h", "herdan_vm", "orlov_z"),
        tl_fs=("entropy", "yule_k", "simpson_d", "hdd"),
        t=("mtld",),
    )
    measure_to_class = {m: c for c, v in classes.items() for m in v}
    func = measures[measure]
    cls = measure_to_class[measure]
    for i in range(int(len(tokens) / window_size)):
        chunk = tokens[i * window_size : (i * window_size) + window_size]
        txt_len, vocab_size, freq_spectrum = preprocess(chunk, fs=True)
        if cls == "tl_vs":
            result = func(txt_len, vocab_size)
        elif cls == "vs_fs":
            result = func(vocab_size, freq_spectrum)
        elif cls == "tl_vs_fs":
            result = func(txt_len, vocab_size, freq_spectrum)
        elif cls == "tl_fs":
            result = func(txt_len, freq_spectrum)
        elif cls == "t":
            result = func(chunk)
        results.append(result)
    if raw:
        return results
    if ci:
        return (np.mean(results), _sttr_ci(results))
    return np.mean(results)
