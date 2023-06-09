from copy import deepcopy
from math import ceil

from numpy.random import Generator

from tsp_as.appointment.heavy_traffic import compute_schedule as compute_ht_schedule
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
        The percentage of clients to remove.
    """
    visits = deepcopy(solution.visits)
    unassigned = []

    num_destroy = ceil(len(solution) * pct_destroy)  # at least one

    for cust in rng.choice(visits, num_destroy, replace=False):
        unassigned.append(cust)
        visits.remove(cust)

    assert len(solution.visits) == len(visits) + num_destroy

    schedule = compute_ht_schedule(visits, solution.data, solution.cost_evaluator)

    return Solution(
        solution.data, solution.cost_evaluator, visits, schedule, unassigned
    )
