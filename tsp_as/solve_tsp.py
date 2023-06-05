from time import perf_counter

import elkai
from alns.Result import Result
from alns.Statistics import Statistics

from tsp_as.classes import Solution


def solve_tsp(seed, data, cost_evaluator, max_iterations=1000, **kwargs):
    """
    Solves the TSP without appointment scheduling.
    """
    if max_iterations is None:
        max_iterations = 1000

    start = perf_counter()
    visits = elkai.solve_float_matrix(data.distances, runs=max_iterations)
    visits.remove(0)  # remove depot

    stats = Statistics()
    stats.collect_objective(0)
    stats.collect_runtime(perf_counter() - start)

    return Result(Solution(data, cost_evaluator, visits), stats)
