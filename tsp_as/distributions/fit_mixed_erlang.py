import math


def fit_mixed_erlang(mean: float, scv: float) -> tuple[int, float, float]:
    """
    Returns the parameters of a fitted mixed Erlang distribution that matches
    the first and second moment of the input distribution.
    """
    # In contrast to the paper, we use (K, K + 1) phases here instead of
    # (K - 1, K) phases.
    K = math.floor(1 / scv)
    prob = ((K + 1) * scv - math.sqrt((K + 1) * (1 - K * scv))) / (scv + 1)
    mu = (K + 1 - prob) / mean

    return K, prob, mu
