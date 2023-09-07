from copy import copy

from ras.appointment.heavy_traffic import compute_schedule as compute_ht_schedule
from ras.classes import CostEvaluator, ProblemData, Solution


def greedy_insert(
    solution: Solution, rng, data: ProblemData, cost_evaluator: CostEvaluator, **kwargs
):
    """
    Insert the unassigned customers into the best place, one-by-one.
    """
    unassigned = copy(solution.unassigned)
    rng.shuffle(unassigned)

    visits = copy(solution.visits)

    while unassigned:
        customer = unassigned.pop()
        best_idx = _best_insert_idx(data, cost_evaluator, visits, customer)
        visits.insert(best_idx, customer)

    return _create_new_ht_solution(data, cost_evaluator, visits)


def _best_insert_idx(
    data: ProblemData, cost_evaluator: CostEvaluator, visits: list[int], customer: int
) -> int:
    """
    Find the best insertion index for the customer in the current visits order.

    We do this by trying all possible insertion points and selecting the one
    that leads to the lowest total cost.
    """
    best_idx = None
    best_cost = float("inf")

    for idx in range(len(visits) + 1):
        new_visits = copy(visits)
        new_visits.insert(idx, customer)

        if _compute_new_travel_cost(data, cost_evaluator, new_visits) > best_cost:
            # If the new travel cost is already higher than the best cost, we
            # can stop the search to avoid expensive idle/wait computations.
            continue

        new = _create_new_ht_solution(data, cost_evaluator, new_visits)

        if best_cost is None or new.cost < best_cost:
            best_cost = new.cost
            best_idx = idx

    assert best_idx is not None  # must always find an insertion index

    return best_idx


def _create_new_ht_solution(
    data: ProblemData, cost_evaluator: CostEvaluator, visits: list[int]
) -> Solution:
    """
    Create a new solution from the given list of client visits by computing
    a heavy traffic schedule.
    """
    ht_schedule = compute_ht_schedule(data, cost_evaluator, visits)
    return Solution(data, cost_evaluator, visits, ht_schedule)


def _compute_new_travel_cost(
    data: ProblemData, cost_evaluator: CostEvaluator, visits: list[int]
) -> float:
    """
    Compute the travel cost of the given visits order.
    """
    distance = data.distances[[0] + visits, visits + [0]].sum()
    return cost_evaluator.travel_weight * distance
