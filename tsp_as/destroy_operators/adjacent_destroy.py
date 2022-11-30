from copy import deepcopy

from tsp_as.classes import Solution


def adjacent_destroy(solution: Solution, rng, n_destroy, **kwargs) -> Solution:
    """
    Randomly remove a number adjacent customers from the solution.
    """
    destroyed = deepcopy(solution)

    start = rng.integers(len(solution.tour) - n_destroy)
    custs_to_remove = [solution.tour[start + idx] for idx in range(n_destroy)]

    for cust in custs_to_remove:
        destroyed.unassigned.append(cust)
        destroyed.tour.remove(cust)

    return destroyed
