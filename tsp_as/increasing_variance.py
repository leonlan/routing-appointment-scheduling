from time import perf_counter

from alns.Result import Result
from alns.Statistics import Statistics

from tsp_as.classes import Solution


def increasing_variance(seed, data, cost_evaluator, **kwargs):
    """
    Creates a visits in increasing order of variances.
    """
    start = perf_counter()

    service_var = data.service_scv * data.service**2
    visits = service_var.argsort().tolist()
    visits.remove(0)  # ignore the depot
    solution = Solution(data, cost_evaluator, visits)

    # This little hack allows us to use the same interface for ALNS-based
    # heuristics and the SCV heuristic.
    stats = Statistics()
    stats.collect_objective(solution.cost)
    stats.collect_runtime(perf_counter() - start)

    return Result(solution, stats)
