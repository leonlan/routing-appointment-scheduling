import math

from numpy.testing import assert_, assert_allclose


def fit_mixed_erlang(mean: float, scv: float) -> tuple[int, float, float]:
    """
    Returns the parameters of a fitted mixed Erlang distribution that matches
    the first and second moment of the input distribution.
    """
    K = math.ceil(1 / scv)
    prob = (K * scv - math.sqrt(K * (1 - (K - 1) * scv))) / (scv + 1)
    mu = (K - prob) / mean  # rate aka inverse scale

    _test_moments(mean, scv, K, prob, mu)

    return K, prob, mu


def _test_moments(mean: float, scv: float, K: int, prob: float, mu: float):
    # Test that the first moment is matched.
    actual_mean = prob * (K - 1) / mu + (1 - prob) * K / mu
    assert_allclose(actual_mean, mean)

    # Test that the second moment is matched.
    variance = (K - prob**2) / (K - prob) ** 2 * actual_mean**2
    actual_second_moment = variance + actual_mean**2
    assert_allclose(actual_second_moment, (scv + 1) * mean**2)

    # Test that probability is between 0 and 1.
    assert_(0 <= prob <= 1)
