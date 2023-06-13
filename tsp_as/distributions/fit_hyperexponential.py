import numpy as np


def fit_hyperexponential(mean: float, scv: float):
    """
    Returns the parameters of a fitted hyperexponential distribution that
    matches the first and second moment of the input distribution.
    """
    prob = 1 / 2 * (1 + np.sqrt((scv - 1) / (scv + 1)))
    mu1 = 2 * prob / mean
    mu2 = 2 * (1 - prob) / mean
    return prob, mu1, mu2
