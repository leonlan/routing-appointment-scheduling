from itertools import permutations
from time import perf_counter

from alns.Result import Result
from alns.Statistics import Statistics

from tsp_as.classes import Solution


def full_enumeration(seed, data):
    """
    Solves using a full enumeration of all possible tours. This is a very
    naive approach, but it is useful for testing purposes. It is also
    useful for comparing the performance of heuristics.

    Parameters
    ----------
    seed : int
        The seed for the random number generator.
    data : ProblemData
        The data for the problem instance.
    """
    start = perf_counter()

    all_sols = [Solution(data, list(tour)) for tour in permutations(range(1, 6))]
    optimal = min(all_sols, key=lambda sol: sol.cost)

    # This little hack allows us to use the same interface for ALNS-based
    # heuristics and the SCV heuristic.
    stats = Statistics()
    stats.collect_objective(optimal.cost)
    stats.collect_runtime(perf_counter() - start)

    return Result(optimal, stats)
