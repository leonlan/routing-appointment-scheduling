import time

import elkai

from tsp_as.appointment.true_optimal import compute_optimal_schedule
from tsp_as.classes import Solution

from .Result import Result


def tsp(seed, data, cost_evaluator, max_iterations=None, **kwargs):
    """
    Solves the appointment scheduling problem by computing a solution to the
    deterministic TSP problem. We compute an optimal TSP tour (only
    considering one orientation) and then compute the optimal schedule.

    Parameters
    ----------
    seed
        The random seed to use for the solver (unused).
    data
        The data object containing the problem data.
    cost_evaluator
        The cost evaluator to use for computing the cost of a schedule.
    max_iterations
        The maximum number of iterations to run the solver for.

    Returns
    -------
    Result
        The algorithm results.
    """
    start = time.perf_counter()

    if max_iterations is None:
        max_iterations = 10000

    visits = elkai.solve_float_matrix(data.distances, runs=max_iterations)
    visits.remove(0)  # remove depot

    schedule = compute_optimal_schedule(data, cost_evaluator, visits)
    solution = Solution(data, cost_evaluator, visits, schedule)
    return Result(solution, time.perf_counter() - start, max_iterations)
