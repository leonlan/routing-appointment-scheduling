from itertools import product

import numpy as np
from numpy.random import Generator

from ras.classes import ProblemData
from ras.distributions import (
    fit_hyperexponential,
    fit_mixed_erlang,
    hyperexponential_rvs,
    mixed_erlang_rvs,
)


def _sample_distance_matrices(
    data: ProblemData, num_samples: int, rng: Generator
) -> np.ndarray:
    """
    Samples a number of distances matrix.

    Parameters
    ----------
    data
        ProblemData object.
    num_samples
        The number of samples.
    rng
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Sampled distances matrix.
    """
    distances = np.zeros((data.dimension, data.dimension, num_samples))

    for i, j in product(range(data.dimension), repeat=2):
        if i == j:
            continue

        mean, scv = data.distances[i, j], data.distances_scv[i, j]

        if scv < 1:  # Mixed Erlang case
            K, p, mu = fit_mixed_erlang(mean, scv)
            distances[i, j, :] = mixed_erlang_rvs(
                [K - 1, K], [1 / mu, 1 / mu], [p, (1 - p)], num_samples, rng
            )

        else:  # Hyperexponential case
            p, mu1, mu2 = fit_hyperexponential(mean, scv)
            distances[i, j, :] = hyperexponential_rvs(
                [1 / mu1, 1 / mu2], [p, (1 - p)], num_samples, rng
            )

    return distances


def _sample_service_times(
    data: ProblemData, num_samples: int, rng: Generator
) -> np.ndarray:
    """
    Samples a number of service times vectors.

    Parameters
    ----------
    data
        ProblemData object.
    num_samples
        The number of samples.
    rng
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Sampled service times.
    """
    service = np.zeros((data.dimension, num_samples))

    for i in range(data.dimension):
        if i == 0:
            continue

        mean, scv = data.service[i], data.service_scv[i]

        if scv < 1:  # Mixed Erlang case
            K, p, mu = fit_mixed_erlang(mean, scv)
            service[i, :] = mixed_erlang_rvs(
                [K - 1, K], [1 / mu, 1 / mu], [p, (1 - p)], num_samples, rng
            )

        else:  # Hyperexponential case
            p, mu1, mu2 = fit_hyperexponential(mean, scv)
            service[i, :] = hyperexponential_rvs(
                [1 / mu1, 1 / mu2], [p, (1 - p)], num_samples, rng
            )

    return service
