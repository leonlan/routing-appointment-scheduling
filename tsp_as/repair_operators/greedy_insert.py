from copy import copy

from tsp_as.appointment.heavy_traffic import compute_schedule as compute_ht_schedule
from tsp_as.classes import Solution


def greedy_insert(solution: Solution, rng, **kwargs):
    """
    Insert the unassigned customers into the best place, one-by-one.
    """
    rng.shuffle(solution.unassigned)

    while solution.unassigned:
        customer = solution.unassigned.pop()
        _opt_insert(solution, customer)

    return solution


def _opt_insert(solution: Solution, customer: int):
    """
    Optimally inserts the customer in the current visits.
    """
    idcs_costs = []

    for idx in range(len(solution.visits) + 1):
        cost = _insert_cost(solution, idx, customer)
        idcs_costs.append((idx, cost))

    idx, _ = min(idcs_costs, key=lambda idx_cost: idx_cost[1])
    solution.insert(idx, customer)


def _insert_cost(solution, idx: int, customer: int) -> float:
    """
    Compute the cost for inserting customer at position idx. The insertion cost
    is the difference between the cost of the current solution and the cost of
    the candidate solution with the inserted customer.
    """
    # We create a copy of the current visits and insert the customer at the
    # specified position. Then we create a new solution object with the
    # candidate visits (which updates the cost) and compute the difference
    # in cost.
    new_visits = copy(solution.visits)
    new_visits.insert(idx, customer)
    new_schedule = compute_ht_schedule(
        new_visits, solution.data, solution.cost_evaluator
    )
    new_solution = Solution(
        solution.data, solution.cost_evaluator, new_visits, new_schedule
    )

    return solution.cost_evaluator(new_solution) - solution.cost_evaluator(solution)
