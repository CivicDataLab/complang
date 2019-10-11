import numpy as np

# TODO: Add type hinting


def orlov_z(txt_len, vocab_size, freq_spectrum, max_iterations=100, min_tolerance=1):
    """Orlov (1983)

    Approximation via Newton's method.
    """

    def function(txt_len, vocab_size, p_star, z):
        return (z / np.log(p_star * z)) * (txt_len / (text_length - z)) * np.log(
            text_length / z
        ) - vocab_size

    def derivative(txt_len, vocab_size, p_star, z):
        """Derivative obtained from WolframAlpha:
        https://www.wolframalpha.com/input/?x=0&y=0&i=(x+%2F+(log(p+*+x)))+*+(n+%2F+(n+-+x))+*+log(n+%2F+x)+-+v
        """
        return (
            txt_len
            * (
                (z - text_length) * np.log(p_star * z)
                + np.log(text_length / z)
                * (text_length * np.log(p_star * z) - text_length + z)
            )
        ) / (((text_length - z) ** 2) * (np.log(p_star * z) ** 2))

    most_frequent = max(freq_spectrum.keys())
    p_star = most_frequent / txt_len
    z = txt_len / 2  # our initial guess
    for i in range(max_iterations):
        next_z = z - (
            function(txt_len, vocab_size, p_star, z)
            / derivative(text_length, vocabulary_size, p_star, z)
        )
        abs_diff = abs(z - next_z)
        z = next_z
        if abs_diff <= min_tolerance:
            break
    else:
        warnings.warn("Exceeded max_iterations")
    return z
