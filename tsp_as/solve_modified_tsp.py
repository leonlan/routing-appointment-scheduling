from itertools import product
from math import exp, factorial, floor, sqrt
from time import perf_counter

import elkai
import numpy as np
from numpy.random import Generator
from numpy.testing import assert_allclose
from scipy.special import gammaincc
from scipy.stats import erlang

from tsp_as.appointment.true_optimal import compute_optimal_schedule
from tsp_as.classes import CostEvaluator, ProblemData, Solution

from .Result import Result

# Number of samples to estimate the CDF
NUM_SAMPLES = 10_000


def solve_modified_tsp(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    max_iterations: int = 10000,
    **kwargs,
):
    """
    Solves the modified TSP algorithm by [1].

    Parameters
    ----------
    seed : int
        Random seed.
    data : Data
        Data object.
    cost_evaluator
        Cost evaluator object.
    max_iterations : int
        Maximum number of iterations.

    Returns
    -------
    Result
        Some results of the run.

    References
    ----------
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

        weight_travel = cost_evaluator.travel_weight
        weight_idle = cost_evaluator.idle_weight
        weight_wait_j = cost_evaluator.wait_weights[j]

        dist = data.distances[i, j]
        appointment_cost = compute_appointment_cost(
            data, weight_idle, weight_wait_j, i, j, rng
        )
        modified_distances[i, j] = dist + (1 / weight_travel) * appointment_cost

    # Solve the TSP using the modified distances
    visits = elkai.solve_float_matrix(modified_distances, runs=max_iterations)
    visits.remove(0)  # remove depot

    # Compute the optimal schedule of the found visits
    schedule = compute_optimal_schedule(visits, data, cost_evaluator)
    solution = Solution(data, cost_evaluator, visits, schedule)

    return Result(solution, perf_counter() - start, 0)


def compute_appointment_cost(
    data: ProblemData,
    weight_idle: float,
    weight_wait: float,
    i: int,
    j: int,
    rng: Generator,
) -> float:
    """
    Computes the appointment cost for an edge (i, j) based on solving the
    newsvendor problem. This is denoted by C_j(x_j) in the paper.

    Parameters
    ----------
    data : Data
        Data object.
    weight_idle : float
        Weight of the idle time.
    weight_wait : float
        Weight of the travel time of customer $j$.
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
    mean, scv = data.arcs_mean[i, j], data.arcs_scv[i, j]

    if scv >= 1:
        # Hyperexponential case
        prob = (1 + np.sqrt((scv - 1) / (scv + 1))) / 2
        mu1 = 2 * prob / mean
        mu2 = 2 * (1 - prob) / mean

        samples = hyperexponential_rvs([mu1, mu2], [prob, (1 - prob)], NUM_SAMPLES, rng)
        appointment_time = compute_appointment_time(samples, weight_wait, weight_idle)
        appointment_cost = cost_hyperexponential(
            appointment_time, prob, mu1, mu2, weight_wait, weight_idle
        )

        assert_allclose(np.mean(samples), mean)
        assert appointment_cost >= 0
    else:
        # Mixed Erlang case
        K = floor(1 / scv)  # Phases are (K, K+1)
        prob = ((K + 1) * scv - sqrt((K + 1) * (1 - K * scv))) / (scv + 1)
        mu = (K + 1 - prob) / mean

        samples = mixed_erlang_rvs(
            [K, K + 1], [K / mu, (K + 1) / mu], [prob, (1 - prob)], NUM_SAMPLES, rng
        )
        appointment_time = compute_appointment_time(samples, weight_wait, weight_idle)
        appointment_cost = cost_mixed_erlang(
            appointment_time, prob, K + 1, mu, weight_wait, weight_idle
        )

        new_mean = prob * K / mu + (1 - prob) * (K + 1) / mu
        assert_allclose(new_mean, mean)
        assert_allclose(np.mean(samples), mean, rtol=0.1)
        assert appointment_cost >= 0

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
    means: list[float],
    weights: list[float],
    num_samples: int,
    rng: Generator,
) -> np.ndarray:
    """
    Generates samples from a mixed Erlang distribution with two phases.

    Parameters
    ----------
    phases : list[int]
        List of phase parameters for each Erlang distribution.
        NOTE this is the same as the shape parameter.
    means : list[float]
        List of loc parameters for each Erlang distribution.
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
    assert len(means) == len(weights), msg

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Select component Erlang distributions based on weights
    choices = rng.choice(len(weights), p=weights, size=num_samples)

    # Generate samples from the selected Erlang distributions
    samples = [
        erlang.rvs(phases[idx], loc=means[idx], scale=0, random_state=rng)
        for idx in choices
    ]

    return np.array(samples)


def compute_appointment_time(
    samples: list[float], weight_wait: float, weight_idle: float
) -> float:
    """
    Computes the optimal inter-appointment time. It is the q-quantile of the
    provided samples, which are either from a hyperexponential or a mixed
    Erlang distribution. This is denoted by x^*_i in the paper.
    """
    q = weight_wait / (weight_wait + weight_idle)

    if not (0 <= q <= 1):
        raise ValueError("q must be between 0 and 1.")

    samples_sorted = np.sort(samples)
    appointment_time = np.percentile(samples_sorted, q * 100)

    assert appointment_time >= 0, "Appointment time must be non-negative."

    return float(appointment_time)


def cost_hyperexponential(x, prob, mu1, mu2, weight_wait, weight_idle):
    expr1 = prob / mu1 * exp(-mu1 * x) + (1 - prob) / mu2 * exp(-mu2 * x)
    expr2 = prob / mu1 + (1 - prob) / mu2

    return (weight_wait + weight_idle) * expr1 + weight_idle * x - weight_idle * expr2


def cost_mixed_erlang(x, p, k, mu, weight_wait, weight_idle):
    expr1 = _mean_mixed_erlang_nonnegative(x, p, k, mu)
    expr2 = (k - 1) / mu + (1 - p) / mu
    return (weight_wait + weight_idle) * expr1 + weight_idle * x - weight_idle * expr2


def _mean_mixed_erlang_nonnegative(x, p, k, mu):
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
