from itertools import product
from math import exp, factorial
from time import perf_counter
from typing import Optional

import elkai
import numpy as np
from numpy.random import Generator
from numpy.testing import assert_, assert_allclose
from scipy.special import gamma, gammaincc

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
NUM_SAMPLES = 500_000


def modified_tsp(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    max_iterations: Optional[int] = None,
    **kwargs,
) -> Result:
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
    schedule = compute_optimal_schedule(data, cost_evaluator, visits)
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
        K, p, mu = fit_mixed_erlang(mean, scv)

        # The mean of each Erlang component is K' / mu, where K' is the number
        # of phases of the Erlang distribution. The scale parameter is
        # computed as mean / K' for each component.
        samples = mixed_erlang_rvs(
            [K - 1, K], [1 / mu, 1 / mu], [p, (1 - p)], NUM_SAMPLES, rng
        )
        appt_time = compute_appointment_time(samples, w_wait, w_idle)
        appt_cost = cost_mixed_erlang(appt_time, p, K, mu, w_wait, w_idle)

        _check_mixed_erlang_samples(appt_time, p, K, mu, samples)

    else:  # Hyperexponential case
        p, mu1, mu2 = fit_hyperexponential(mean, scv)
        samples = hyperexponential_rvs(
            [1 / mu1, 1 / mu2], [p, (1 - p)], NUM_SAMPLES, rng
        )

        appt_time = compute_appointment_time(samples, w_wait, w_idle)
        appt_cost = cost_hyperexponential(appt_time, p, mu1, mu2, w_wait, w_idle)

        _check_hyperexponential_samples(appt_time, p, mu1, mu2, samples)

    _check_moments_samples(mean, scv, samples)
    _check_nonnegative_appt_values(appt_time, appt_cost)

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
    expr1 = _mean_hyperexponential_nonnegative(x, prob, mu1, mu2)
    expr2 = prob / mu1 + (1 - prob) / mu2

    return (weight_wait + weight_idle) * expr1 + weight_idle * x - weight_idle * expr2


def _mean_hyperexponential_nonnegative(x, prob, mu1, mu2):
    return prob / mu1 * exp(-mu1 * x) + (1 - prob) / mu2 * exp(-mu2 * x)


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

    # Gammaincc has an extra factor 1/gamma(k-1), which we don't need in our
    # definition of the upper incomplete Gamma function.
    expr2 = gammaincc(k - 1, mu * x) * gamma(k - 1)

    expr3 = (k - p) / (mu * factorial(k - 1))
    expr4 = (mu * x) ** (k - 1) * exp(-mu * x)

    return expr1 * expr2 + expr3 * expr4


def _check_hyperexponential_samples(x, p, mu1, mu2, samples):
    """
    Verify that the sample means correspond to the theoretical result.
    """
    samples = np.array(samples)

    wait_ = np.mean(np.maximum(0, samples - x))
    true_wait = _mean_hyperexponential_nonnegative(x, p, mu1, mu2)
    assert_allclose(wait_, true_wait, rtol=0.01)

    idle_ = np.mean(np.maximum(0, x - samples))
    mean = p / mu1 + (1 - p) / mu2
    true_idle = true_wait + x - mean
    assert_allclose(idle_, true_idle, rtol=0.01)


def _check_mixed_erlang_samples(x, p, k, mu, samples):
    """
    Verify that the sample means correspond to the theoretical result.
    """
    samples = np.array(samples)

    wait_ = np.mean(np.maximum(0, samples - x))
    true_wait = _mean_mixed_erlang_nonnegative(x, p, k, mu)
    msg = "Sampled wait time does not match theoretical result."
    assert_allclose(wait_, true_wait, rtol=0.01, err_msg=msg)

    idle_ = np.mean(np.maximum(0, x - samples))
    mean = p * (k - 1) / mu + (1 - p) * k / mu
    true_idle = true_wait + x - mean
    msg = "Sampled idle time does not match theoretical result."
    assert_allclose(idle_, true_idle, rtol=0.01, err_msg=msg)


def _check_moments_samples(mean, scv, samples):
    msg = "Mean of samples is not equal to mean of random variable."
    assert_allclose(np.mean(samples), mean, rtol=0.01, err_msg=msg)

    msg = "Second moment of samples is not equal to second moment of random variable."
    moment2 = (scv + 1) * mean**2
    assert_allclose(np.mean(np.power(samples, 2)), moment2, rtol=0.05, err_msg=msg)


def _check_nonnegative_appt_values(appt_time, appt_cost):
    assert_(appt_time >= 0, msg="appt_time must be non-negative.")
    assert_(appt_cost >= 0, msg="appt_cost must be non-negative.")
