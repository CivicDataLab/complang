from typing import List
import itertools
import numpy as np
from measures.len_based import average_token_length_characters
from measures.len_based import average_token_length_syllables
from aggregator import bootstrap


def get(tokens: List[str]):
    tokens = list(itertools.chain.from_iterable(tokens))
    lexical = [
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
        "sichel_s",
        "michea_m",
        "honore_h",
        "herdan_vm",
        "entropy",
        "yule_k",
        "simpson_d",
        "hdd",
        "mtld",
    ]
    word_length = [average_token_length_characters, average_token_length_syllables]
    word_length_names = [
        "average_token_length_characters",
        "average_token_length_syllables",
    ]
    for measure in lexical:
        score, ci = bootstrap(tokens, measure=measure, window_size=3, ci=True)
    for wl, wl_name in zip(word_length, word_length_names):
        score, stdev = wl(tokens)
