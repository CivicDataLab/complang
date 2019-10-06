from typing import List
import numpy as np


def _sttr_ci(results: List[int]) -> np.float64:
    """calculate the confidence interval for sttr """
    return 1.96 * np.std(results) / np.sqrt(len(results))


def preprocess(tokens: List[str], fs: bool=False) -> float:
    """Return text length, vocabulary size and optionally the frequency
    spectrum.

    :param fs: additionally calculate and return the frequency
               spectrum
    """
    text_length = len(tokens)
    vocabulary_size = len(set(tokens))
    if fs:
        frequency_list = collections.Counter(tokens)
        frequency_spectrum = dict(collections.Counter(frequency_list.values()))
        return text_length, vocabulary_size, frequency_spectrum
    return text_length, vocabulary_size


def bootstrap(tokens: List[str], measure: str='type_token_ratio', window_size: int=3, ci: bool=False, raw: bool=False) -> np.float64:
    """calculate bootstrap for lex diversity measures
    as explained in Evert et al. 2017. if measure='type_token_ratio' it calculates
    standardized type-token ratio
    :param ci:  additionally calculate and return the confidence interval, returns a tuple
    :param raw:  return the raw results
    """
    results = []
    measures = dict(type_token_ratio=type_token_ratio,
                    guiraud_r=guiraud_r, herdan_c=herdan_c,
                    dugast_k=dugast_k, maas_a2=maas_a2,
                    dugast_u=dugast_u, tuldava_ln=tuldava_ln,
                    brunet_w=brunet_w, cttr=cttr, summer_s=summer_s,
                    sichel_s=sichel_s, michea_m=michea_m,
                    honore_h=honore_h, entropy=entropy, yule_k=yule_k,
                    simpson_d=simpson_d, herdan_vm=herdan_vm, hdd=hdd,
                    orlov_z=orlov_z, mtld=mtld)
    # tl_vs: text_length, vocabulary_size
    # vs_fs: vocabulary_size, frequency_spectrum
    # tl_vs_fs: text_length, vocabulary_size, frequency_spectrum
    # tl_fs: text_length, frequency_spectrum
    # t: tokens
    classes = dict(tl_vs=("type_token_ratio", "guiraud_r", "herdan_c",
                          "dugast_k", "maas_a2", "dugast_u",
                          "tuldava_ln", "brunet_w", "cttr",
                          "summer_s"),
                   vs_fs=("sichel_s", "michea_m"),
                   tl_vs_fs=("honore_h", "herdan_vm", "orlov_z"),
                   tl_fs=("entropy", "yule_k", "simpson_d", "hdd"),
                   t=("mtld",))
    measure_to_class = {m: c for c, v in classes.items() for m in v}
    func = measures[measure]
    cls = measure_to_class[measure]

    for i in range(int(len(tokens) / window_size)):  # ignore last partial chunk
        chunk = tokens[i * window_size:(i * window_size) + window_size]
        text_length, vocabulary_size, frequency_spectrum = preprocess(chunk, fs=True)
        if cls == "tl_vs":
            result = func(text_length, vocabulary_size)
        elif cls == "vs_fs":
            result = func(vocabulary_size, frequency_spectrum)
        elif cls == "tl_vs_fs":
            result = func(text_length, vocabulary_size, frequency_spectrum)
        elif cls == "tl_fs":
            result = func(text_length, frequency_spectrum)
        elif cls == "t":
            result = func(chunk)
        results.append(result)
    if raw:
        return results
    if ci:
        return (np.mean(results), _sttr_ci(results))
    return np.mean(results)
