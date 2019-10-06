import numpy as np
import scipy


def sichel_s(vocab_size, freq_spectrum):
    """Sichel (1975)"""
    return freq_spectrum.get(2, 0) / vocab_size


def michea_m(vocab_size, freq_spectrum):
    """Michéa (1969, 1971)"""
    return vocab_size / freq_spectrum.get(2, 0)


def honore_h(txt_len, vocab_size, freq_spectrum):
    """Honoré (1979)"""
    return 100 * (np.log(txt_len) / (1 - ((freq_spectrum.get(1, 0)) / (vocab_size))))


def entropy(txt_len, freq_spectrum):
    """"""
    return sum((freq_size * (- np.log(freq / txt_len)) * (freq / txt_len) for freq, freq_size in freq_spectrum.items()))


def yule_k(txt_len, freq_spectrum):
    """Yule (1944)"""
    return 10000 * (sum((freq_size * (freq / txt_len) ** 2 for freq, freq_size in freq_spectrum.items())) - (1 / txt_len))


def simpson_d(txt_len, freq_spectrum):
    """"""
    return sum((freq_size * (freq / txt_len) * ((freq - 1) / (txt_len - 1)) for freq, freq_size in freq_spectrum.items()))


def herdan_vm(txt_len, vocab_size, freq_spectrum):
    """Herdan (1955)"""
    return np.sqrt(sum((freq_size * (freq / txt_len) ** 2 for freq, freq_size in freq_spectrum.items())) - (1 / vocab_size))


def hdd(txt_len, freq_spectrum, sample_size=42):
    """McCarthy and Jarvis (2010)"""
    return sum(((1 - scipy.stats.hypergeom.pmf(0, txt_len, freq, sample_size)) / sample_size for word, freq in freq_spectrum.items()))
