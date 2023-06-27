import time

import elkai

from tsp_as.appointment.true_optimal import compute_optimal_schedule
from tsp_as.classes import Solution

from .Result import Result


def double_orientation_tsp(seed, data, cost_evaluator, max_iterations=None, **kwargs):
    """
    Solves the appointment scheduling problem by computing a solution to the
    deterministic TSP problem. For each orientation of the TSP tour, the
    optimal schedule is computed and the best of the two is returned.

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

    # Try both orientations of the client visits
    schedule1 = compute_optimal_schedule(visits, data, cost_evaluator)
    first = Solution(data, cost_evaluator, visits, schedule1)

    reverse_visits = visits[::-1]
    schedule2 = compute_optimal_schedule(reverse_visits, data, cost_evaluator)
    second = Solution(data, cost_evaluator, reverse_visits, schedule2)

    best = first if first.cost < second.cost else second
    return Result(best, time.perf_counter() - start, max_iterations)
