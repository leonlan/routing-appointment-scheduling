import numpy as np


def objht(x, B, omega_bar):
    """
    Computes the objective value using heavy traffic approximation.
    See (3) in draft.

    TODO
    - What is B (or S in the text)?
    - Why is the regular omega not included?
    """
    sum_sqrt_s = sum([np.sqrt(B[i]) for i in range(len(x))])
    coeff = np.sqrt(2 * omega_bar * (1 - omega_bar))
    return coeff * sum_sqrt_s
