from itertools import product
from math import exp, factorial
from time import perf_counter

import elkai
import numpy as np
from alns.Result import Result
from alns.Statistics import Statistics
from numpy.random import Generator

from tsp_as.classes import Solution


def solve_modified_tsp(seed, data, max_iterations=1000, **kwargs):
    """
    Solves the modified TSP algorithm by [1].
    """
    start = perf_counter()

    NUM_SAMPLES = 1_000_000
    rng = np.random.default_rng(seed)

    # Compute the normalized "newsvendor appointment costs" for each edge.
    modified_distances = np.zeros_like(data.distances)

    for i, j in product(range(data.dimension), repeat=2):
        if i == j:  # ignore self-loops
            continue

        quantile = data.omega_wait / (data.omega_wait + data.omega_idle)

        if data.scvs[i, j] >= 1:
            # Hyperexponential case
            samples = hyperexponential_rvs([10, 5], [1, 1], rng, NUM_SAMPLES)
            x = find_quantile(samples, quantile)
            cost = cost_hyperexponential(
                x, alpha, mu1, mu2, data.omega_weight, data.omega_idle
            )
        else:
            # Mixed Erlang case
            samples = mixed_erlang_rvs([10, 5], [1, 1], [1, 1], rng, NUM_SAMPLES)
            x = find_quantile(samples, quantile)
            cost = cost_mixed_erlang(x, p, k, mu, data.omega_weight, data.omega_idle)

        old_dist = data.distances[i, j]
        normalization = 1 / data.omega_travel * cost

        modified_distances[i, j] = old_dist + normalization

    # Solve the TSP using the modified distances
    tour = elkai.solve_float_matrix(modified_distances, runs=max_iterations)
    tour.remove(0)  # remove depot

    stats = Statistics()
    stats.collect_objective(0)
    stats.collect_runtime(perf_counter() - start)

    return Result(Solution(data, tour), stats)


def mixed_erlang_rvs(
    shapes: list[float],
    scales: list[float],
    weights: list[float],
    rng: Generator,
    num_samples=1,
) -> np.ndarray[float]:
    msg = "Input lists must have the same length."
    assert len(shapes) == len(scales) == len(weights), msg

    size = len(shapes)

    shapes = np.array(shapes)
    scales = np.array(scales)
    weights = np.array(weights)

    # Normalize weights
    weights = weights / weights.sum()

    # Select component Erlang distributions based on weights
    components = rng.choice(size, size=num_samples, p=weights)

    # Generate samples from the selected Erlang distributions
    samples = [rng.gamma(shapes[k], scales[k]) for k in components]

    return np.array(samples)


def hyperexponential_rvs(
    rates: list[float], weights: list[float], rng: Generator, num_samples=1
) -> np.ndarray[float]:
    msg = "Input lists must have the same length."
    assert len(rates) == len(weights), msg

    # Convert input lists into NumPy arrays for easier manipulation
    rates = np.array(rates)
    weights = np.array(weights)

    # Normalize weights
    weights = weights / weights.sum()

    # Select component exponential distributions based on weights
    components = rng.choice(len(rates), size=num_samples, p=weights)

    # Generate samples from the selected exponential distributions
    samples = [rng.exponential(1 / rates[k]) for k in components]

    return samples


def find_quantile(samples: list[float], q: float) -> float:
    if not (0 <= q <= 1):
        raise ValueError("q must be between 0 and 1.")

    samples_sorted = np.sort(samples)
    return np.percentile(samples_sorted, q * 100)


def cost_hyperexponential(x, alpha, mu1, mu2, omega_weight, omega_idle):
    expr1 = alpha / mu1 * exp(-mu1 * x) + (1 - alpha) / mu2 * exp(-mu2 * x)
    expr2 = alpha / mu1 + (1 - alpha) / mu2

    return (omega_weight + omega_idle) * expr1 + omega_idle * x - omega_idle * expr2


def cost_mixed_erlang(x, p, k, mu, omega_weight, omega_idle):
    expr1 = mean_mixed_erlang_nonnegative(x, p, k, mu)
    expr2 = (k - 1) / mu + (1 - p) / mu
    return (omega_weight + omega_idle) * expr1 + omega_idle * x - omega_idle * expr2


def mean_mixed_erlang_nonnegative(x, p, k, mu):
    """
    Computes the mean of a non-negative mixed Erlang distribution, specifically:

        E[X - c]^+

    where X is a mixed Erlang distribution with parameters p, k, mu and c is a
    constant.
    """
    expr1 = (k - p - mu * x) / (mu * factorial(k - 2))
    expr2 = gamma(k - 1, mu * x)  # TODO this gamma is not well defined

    expr3 = (k - p) / (mu * factorial(k - 1))
    expr4 = (mu * x) ** (k - 1) * exp(-mu * x)

    return expr1 * expr2 + expr3 * expr4
