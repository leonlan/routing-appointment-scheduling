from copy import deepcopy
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

    enum_data = deepcopy(data)
    enum_data.objective = "to"

    perms = permutations(range(1, data.dimension))
    all_sols = [Solution(enum_data, list(tour)) for tour in perms]
    optimal = min(all_sols, key=lambda sol: sol.cost)
    print(optimal.tour)

    # This little hack allows us to use the same interface for ALNS-based
    # heuristics and the SCV heuristic.
    stats = Statistics()
    stats.collect_objective(optimal.cost)
    stats.collect_runtime(perf_counter() - start)

    return Result(optimal, stats)
