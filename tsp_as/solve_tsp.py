import time

import elkai

from tsp_as.appointment.true_optimal import compute_optimal_schedule
from tsp_as.classes import Solution

from .Result import Result


def solve_tsp(seed, data, cost_evaluator, max_iterations=None, **kwargs):
    """
    Solves the TSP without appointment scheduling.
    """
    start = time.perf_counter()

    if max_iterations is None:
        max_iterations = 10000

    visits = elkai.solve_float_matrix(data.distances, runs=max_iterations)
    visits.remove(0)  # remove depot

    schedule = compute_optimal_schedule(visits, data, cost_evaluator)
    solution = Solution(data, cost_evaluator, visits, schedule)

    return Result(solution, time.perf_counter() - start, max_iterations)
