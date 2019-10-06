import numpy as np

# TODO: add type hinting

def mattr(tokens, window_size=1000):
    """Calculate the Moving-Average Type-Token Ratio (Covington and
    McFall, 2010).

    M.A. Covington, J.D. McFall: Cutting the Gordon Knot. In: Journal
    of Quantitative Linguistics 17,2 (2010), p. 94-100. DOI:
    10.1080/09296171003643098
    """
    ttr_values = []
    window_frequencies = collections.Counter(tokens[0:window_size])
    for window_start in range(1, len(tokens) - (window_size + 1)):
        window_end = window_start + window_size
        word_to_pop = tokens[window_start - 1]
        window_frequencies[word_to_pop] -= 1
        window_frequencies[tokens[window_end]] += 1
        if window_frequencies[word_to_pop] == 0:
            del window_frequencies[word_to_pop]
        # type-token ratio for the current window:
        ttr_values.append(len(window_frequencies) / window_size)
    return np.mean(ttr_values)


def mtld(tokens, factor_size=0.72):
    """Implementation following the description in McCarthy and Jarvis
    (2010).
    """
    def _mtld(tokens, factor_size, reverse=False):
        factors = 0
        factor_lengths = []
        types = set()
        token_count = 0
        token_iterator = iter(tokens)
        if reverse:
            token_iterator = reversed(tokens)
        for token in token_iterator:
            types.add(token)
            token_count += 1
            if len(types) / token_count <= factor_size:
                factors += 1
                factor_lengths.append(token_count)
                types = set()
                token_count = 0
        if token_count > 0:
            ttr = len(types) / token_count
            factors += (1 - ttr) / (1 - factor_size)
            factor_lengths.append(token_count)
        return len(tokens) / factors
    forward_mtld = _mtld(tokens, factor_size)
    reverse_mtld = _mtld(tokens, factor_size, reverse=True)
    return np.mean((forward_mtld, reverse_mtld))

