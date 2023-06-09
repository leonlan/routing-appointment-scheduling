from itertools import permutations
from time import perf_counter

from alns.Result import Result
from alns.Statistics import Statistics

from tsp_as.classes import Solution


def full_enumeration(seed, data, cost_evaluator, **kwargs):
    """
    Solves using a full enumeration of all possible visitss. This is a very
    naive approach, but it is useful for testing purposes. It is also
    useful for comparing the performance of heuristics.

    Parameters
    ----------
    seed
        The seed for the random number generator.
    data
        The data for the problem instance.
    cost_evaluator
        The cost evaluator.
    """
    # TODO rewrite
    start = perf_counter()

    perms = permutations(range(1, data.dimension))
    all_sols = [Solution(data, cost_evaluator, list(visits)) for visits in perms]
    optimal = min(all_sols, key=lambda sol: sol.objective())

    # This little hack allows us to use the same interface for ALNS-based
    # heuristics and the SCV heuristic.
    stats = Statistics()
    stats.collect_objective(optimal.objective())
    stats.collect_runtime(perf_counter() - start)

    return Result(optimal, stats)
