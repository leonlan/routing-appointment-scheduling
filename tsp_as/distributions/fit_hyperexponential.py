import numpy as np
from numpy.testing import assert_allclose


def fit_hyperexponential(mean: float, scv: float):
    """
    Returns the parameters of a fitted hyperexponential distribution that
    matches the first and second moment of the input distribution.
    """
    prob = 1 / 2 * (1 + np.sqrt((scv - 1) / (scv + 1)))
    mu1 = 2 * prob / mean
    mu2 = 2 * (1 - prob) / mean

    _test_moments(mean, scv, prob, mu1, mu2)

    return prob, mu1, mu2


def _test_moments(mean: float, scv: float, prob: float, mu1: float, mu2: float):
    # Test that first moment matched.
    new_mean = (prob / mu1) + ((1 - prob) / mu2)
    assert_allclose(new_mean, mean)

    # That that second moment matched.
    # Second moment is given by E[X^2] = SCV(X) * E[X]^2 + E[X]^2
    second_moment = (prob / mu1**2) + ((1 - prob) / mu2**2)
    second_moment *= 2
    assert_allclose(second_moment, (scv + 1) * mean**2)
