from itertools import product
from math import exp, factorial, floor, sqrt
from time import perf_counter

import elkai
import numpy as np
from alns.Result import Result
from alns.Statistics import Statistics
from numpy.random import Generator
from scipy.special import gammaincc
from scipy.stats import erlang

from tsp_as.classes import ProblemData, Solution

# Number of samples to estimate the CDF
NUM_SAMPLES = 1_000_000
NUM_SAMPLES = 1000


def solve_modified_tsp(seed, data, max_iterations=1000, **kwargs):
    """
    Solves the modified TSP algorithm by [1].

    Parameters
    ----------
    seed : int
        Random seed.
    data : Data
        Data object.
    max_iterations : int
        Maximum number of iterations.

    Returns
    -------
    Result
        Some results of the run.

    [1]: Zhan, Y., Wang, Z., & Wan, G. (2021). Home service routing and
    appointment scheduling with stochastic service times. European Journal of
    Operational Research, 288(1), 98–110.
    """
    start = perf_counter()

    rng = np.random.default_rng(seed)

    # Compute the modified distances by adding to each edge the appointment
    # cost approximated as newsvendor problem.
    modified_distances = np.zeros_like(data.distances)

    for i, j in product(range(data.dimension), repeat=2):
        if i == j:  # ignore self-loops
            continue

        dist = data.distances[i, j]
        appointment_cost = compute_appointment_cost(data, i, j, rng)

        modified_distances[i, j] = dist + (1 / data.omega_travel) * appointment_cost

    # Solve the TSP using the modified distances
    tour = elkai.solve_float_matrix(modified_distances, runs=max_iterations)
    tour.remove(0)  # remove depot

    stats = Statistics()
    stats.collect_objective(0)
    stats.collect_runtime(perf_counter() - start)

    return Result(Solution(data, tour), stats)


def compute_appointment_cost(
    data: ProblemData, i: int, j: int, rng: Generator
) -> float:
    """
    Computes the appointment cost for an edge (i, j) based on solving the
    newsvendor problem.

    Parameters
    ----------
    data : Data
        Data object.
    i : int
        Index of the first node.
    j : int
        Index of the second node.
    rng : Generator
        NumPy random number generator.

    Returns
    -------
    float
        Appointment cost.
    """
    mean, scv = data.means[i, j], data.scvs[i, j]
    quantile = data.omega_wait / (data.omega_wait + data.omega_idle)

    if scv >= 1:
        # Hyperexponential case
        prob = (1 + np.sqrt((scv - 1) / (scv + 1))) / 2
        mu1 = 2 * prob / mean
        mu2 = 2 * (1 - prob) / mean

        samples = hyperexponential_rvs([mu1, mu2], [prob, (1 - prob)], NUM_SAMPLES, rng)
        appointment_time = find_quantile(samples, quantile)
        appointment_cost = cost_hyperexponential(
            appointment_time, prob, mu1, mu2, data.omega_wait, data.omega_idle
        )
    else:
        # Mixed Erlang case
        K = floor(1 / scv)
        prob = ((K + 1) * scv - sqrt((K + 1) * (1 - K * scv))) / (scv + 1)
        scale = (K + 1 - prob) / mean  # scale

        samples = mixed_erlang_rvs(
            [K - 1, K], [scale, scale], [prob, (1 - prob)], NUM_SAMPLES, rng
        )
        appointment_time = find_quantile(samples, quantile)
        appointment_cost = cost_mixed_erlang(
            appointment_time, prob, K, scale, data.omega_wait, data.omega_idle
        )

    return appointment_cost


def hyperexponential_rvs(
    scales: list[float], weights: list[float], num_samples: int, rng: Generator
) -> np.ndarray[float]:
    """
    Generates samples from a hyperexponential distribution consisting of two
    exponential distributions.

    Parameters
    ----------
    scales : list[float]
        List of scale (mean) parameters for each exponential distribution.
    weights : list[float]
        List of weights (probabilities) for each exponential distribution.
    num_samples : int
        Number of samples to generate.
    rng : Generator
        NumPy random number generator.

    Returns
    -------
    np.ndarray[float]
        Array of samples from the hyperexponential distribution.
    """
    msg = "Input lists must have the same length."
    assert len(scales) == len(weights), msg

    # Convert input lists into NumPy arrays for easier manipulation
    scales = np.array(scales)
    weights = np.array(weights)

    # Normalize weights to probabilities
    weights = weights / weights.sum()

    # Select component exponential distributions based on weights
    components = rng.choice(len(weights), size=num_samples, p=weights)

    # Generate samples from the selected exponential distributions
    samples = [rng.exponential(scales[k]) for k in components]

    return samples


def mixed_erlang_rvs(
    phases: list[int],
    scales: list[float],
    weights: list[float],
    num_samples: int,
    rng: Generator,
) -> np.ndarray[float]:
    """
    Generates samples from a mixed Erlang distribution with two phases.

    Parameters
    ----------
    phases : list[int]
        List of phase parameters for each Erlang distribution.
    scales : list[float]
        List of scale parameters for each Erlang distribution.
    weights : list[float]
        List of weights (probabilities) for each Erlang distribution.
    num_samples : int
        Number of samples to generate.
    rng : Generator
        NumPy random number generator.

    Returns
    -------
    np.ndarray[float]
        Array of samples from the mixed Erlang distribution.
    """
    msg = "Input lists must have the same length."
    assert len(scales) == len(weights), msg

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Select component Erlang distributions based on weights
    choices = rng.choice(len(weights), size=num_samples, p=weights)

    # Generate samples from the selected Erlang distributions
    samples = [
        erlang.rvs(phases[k], scale=scales[k], random_state=rng) for k in choices
    ]

    return np.array(samples)


def find_quantile(samples: list[float], q: float) -> float:
    if not (0 <= q <= 1):
        raise ValueError("q must be between 0 and 1.")

    samples_sorted = np.sort(samples)
    return np.percentile(samples_sorted, q * 100)


def cost_hyperexponential(x, prob, mu1, mu2, omega_wait, omega_idle):
    expr1 = prob / mu1 * exp(-mu1 * x) + (1 - prob) / mu2 * exp(-mu2 * x)
    expr2 = prob / mu1 + (1 - prob) / mu2

    return (omega_wait + omega_idle) * expr1 + omega_idle * x - omega_idle * expr2


def cost_mixed_erlang(x, p, k, mu, omega_wait, omega_idle):
    expr1 = mean_mixed_erlang_nonnegative(x, p, k, mu)
    expr2 = (k - 1) / mu + (1 - p) / mu
    return (omega_wait + omega_idle) * expr1 + omega_idle * x - omega_idle * expr2


def mean_mixed_erlang_nonnegative(x, p, k, mu):
    """
    Computes the mean of a non-negative mixed Erlang distribution, specifically:

        E[X - c]^+

    where X is a mixed Erlang distribution with parameters p, k, mu and c is a
    constant.
    """
    expr1 = (k - p - mu * x) / (mu * factorial(k - 2))

    expr2 = gammaincc(k - 1, mu * x)  # Regularized upper incomplete Gamma

    expr3 = (k - p) / (mu * factorial(k - 1))
    expr4 = (mu * x) ** (k - 1) * exp(-mu * x)

    return expr1 * expr2 + expr3 * expr4
