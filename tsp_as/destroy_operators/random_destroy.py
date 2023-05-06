from copy import deepcopy
from math import ceil

from numpy.random import Generator

from tsp_as.classes import Solution


def random_destroy(
    solution: Solution, rng: Generator, pct_destroy: float = 0.15, **kwargs
) -> Solution:
    """
    Randomly removes clients from the solution.

    Parameters
    ----------
    solution
        The solution to destroy.
    rng
        The random number generator.
    pct_destroy
        The percentage of clients to remove, by default 0.2.
    """
    destroyed = deepcopy(solution)
    num_destroy = ceil(len(solution) * pct_destroy)  # at least one

    for cust in rng.choice(destroyed.tour, num_destroy, replace=False):
        destroyed.unassigned.append(cust)
        destroyed.tour.remove(cust)

    # After removing customers, we need to update the solution's costs
    # because the costs are cached to minimize computations.
    destroyed.update()

    assert len(solution.tour) == len(destroyed.tour) + num_destroy

    return destroyed
