from time import perf_counter

import elkai
from alns.Result import Result
from alns.Statistics import Statistics

from tsp_as.classes import Solution


def solve_tsp(seed, data, max_iterations=1000, **kwargs):
    """
    Solves the TSP without appointment scheduling.
    """
    start = perf_counter()
    tour = elkai.solve_float_matrix(data.distances, runs=max_iterations)
    tour.remove(0)  # remove depot

    stats = Statistics()
    stats.collect_objective(0)
    stats.collect_runtime(perf_counter() - start)

    return Result(Solution(data, tour), stats)
