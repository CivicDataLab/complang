from typing import List
import numpy as np


def _sttr_ci(results: List[int]) -> np.float64:
    """calculate the confidence interval for sttr """
    return 1.96 * np.std(results) / np.sqrt(len(results))
