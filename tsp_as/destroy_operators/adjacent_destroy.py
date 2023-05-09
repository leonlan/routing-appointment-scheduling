from copy import deepcopy
from math import ceil

from numpy.random import Generator

from tsp_as.classes import Solution


def adjacent_destroy(
    solution: Solution, rng: Generator, pct_destroy: float = 0.15, **kwargs
) -> Solution:
    """
    Randomly removes a number adjacent customers from the solution.

    Parameters
    ----------
    solution
        The solution to destroy.
    rng
        The random number generator.
    pct_destroy
        The percentage of customers to remove.
    """
    destroyed = deepcopy(solution)
    num_destroy = ceil(len(solution) * pct_destroy)  # at least one

    start = rng.integers(len(solution.tour) - num_destroy)
    custs_to_remove = [solution.tour[start + idx] for idx in range(num_destroy)]

    for cust in custs_to_remove:
        destroyed.unassigned.append(cust)
        destroyed.tour.remove(cust)

    destroyed.update()  # Update the solution's costs

    assert len(solution.tour) == len(destroyed.tour) + num_destroy

    return destroyed
