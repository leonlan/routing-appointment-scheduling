from copy import deepcopy

from numpy.random import Generator

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

    num_destroy = rng.integers(max_num_destroy) + 1  # destroy at least 1 cust
    num_destroy = min(num_destroy, len(visits) - 1)  # keep at least 1 cust

    for cust in rng.choice(visits, num_destroy, replace=False):
        unassigned.append(cust)
        visits.remove(cust)

    assert len(solution.visits) == len(visits) + num_destroy

    schedule = compute_ht_schedule(visits, data, cost_evaluator)

    return Solution(data, cost_evaluator, visits, schedule, unassigned)
