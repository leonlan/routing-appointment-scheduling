from copy import deepcopy

from numpy.random import Generator
from numpy.testing import assert_equal

from tsp_as.appointment.heavy_traffic import compute_schedule as compute_ht_schedule
from tsp_as.classes import CostEvaluator, ProblemData, Solution


def random_destroy(
    solution: Solution,
    rng: Generator,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    max_num_destroy: int = 5,
    **kwargs
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

    num_destroy = rng.integers(1, min(max_num_destroy, len(visits)))

    for cust in rng.choice(visits, num_destroy, replace=False):
        unassigned.append(cust)
        visits.remove(cust)

    assert_equal(len(solution.visits), len(visits) + num_destroy)

    schedule = compute_ht_schedule(visits, data, cost_evaluator)

    return Solution(data, cost_evaluator, visits, schedule, unassigned)
