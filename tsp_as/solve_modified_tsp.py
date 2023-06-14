from itertools import product
from math import exp, factorial
from time import perf_counter
from typing import Optional

import elkai
import numpy as np
from numpy.random import Generator
from numpy.testing import assert_, assert_allclose
from scipy.special import gammaincc

from tsp_as.appointment.true_optimal import compute_optimal_schedule
from tsp_as.classes import CostEvaluator, ProblemData, Solution
from tsp_as.distributions import (
    fit_hyperexponential,
    fit_mixed_erlang,
    hyperexponential_rvs,
    mixed_erlang_rvs,
)

from .Result import Result

# Number of samples to estimate the CDF
NUM_SAMPLES = 5_000


def solve_modified_tsp(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    max_iterations: Optional[int] = None,
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

    if max_iterations is None:
        max_iterations = 10000

    rng = np.random.default_rng(seed)

    # Compute the modified distances by adding to each edge the appointment
    # cost approximated as newsvendor problem.
    modified_distances = np.zeros_like(data.distances)

    for i, j in product(range(data.dimension), repeat=2):
        if i == j:  # ignore self-loops
            continue

        w_travel = cost_evaluator.travel_weight
        w_idle = cost_evaluator.idle_weight
        w_wait_j = cost_evaluator.wait_weights[j]

        dist = data.distances[i, j]
        appointment_cost = compute_appointment_cost(data, w_wait_j, w_idle, i, j, rng)
        modified_distances[i, j] = dist + (1 / w_travel) * appointment_cost

    # Solve the TSP using the modified distances
    visits = elkai.solve_float_matrix(modified_distances, runs=max_iterations)
    visits.remove(0)  # remove depot

    # Compute the optimal schedule of the found visits
    schedule = compute_optimal_schedule(visits, data, cost_evaluator)
    solution = Solution(data, cost_evaluator, visits, schedule)

    return Result(solution, perf_counter() - start, 0)


def compute_appointment_cost(
    data: ProblemData,
    w_wait: float,
    w_idle: float,
    i: int,
    j: int,
    rng: Generator,
) -> float:
    """
    Computes the appointment cost for an edge (i, j) based on solving the
    newsvendor problem. This is denoted by C_j(x_j) in the paper.

    Parameters
    ----------
    data
        ProblemData object.
    w_idle
        Weight of the idle time.
    w_wait
        Weight of the travel time of customer $j$.
    i
        Index of the first node.
    j
        Index of the second node.
    rng
        NumPy random number generator.

    Returns
    -------
    float
        Appointment cost.
    """
    mean, scv = data.arcs_mean[i, j], data.arcs_scv[i, j]

    if scv < 1:  # Mixed Erlang case
        K, prob, mu = fit_mixed_erlang(mean, scv)
        samples = mixed_erlang_rvs(
            [K - 1, K], [(K - 1) / mu, K / mu], [prob, (1 - prob)], NUM_SAMPLES, rng
        )

        appt_time = compute_appointment_time(samples, w_wait, w_idle)
        appt_cost = cost_mixed_erlang(appt_time, prob, K, mu, w_wait, w_idle)
    else:  # Hyperexponential case
        prob, mu1, mu2 = fit_hyperexponential(mean, scv)
        samples = hyperexponential_rvs(
            [1 / mu1, 1 / mu2], [prob, (1 - prob)], NUM_SAMPLES, rng
        )

        appt_time = compute_appointment_time(samples, w_wait, w_idle)
        appt_cost = cost_hyperexponential(appt_time, prob, mu1, mu2, w_wait, w_idle)

    # Check that the mean of the samples is equal to the mean of the random varialbe
    # and that the appointment time and cost is non-negative.
    assert_allclose(np.mean(samples), mean, rtol=0.1)
    assert_(appt_time >= 0)
    assert_(appt_cost >= 0)

    return appt_cost


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
