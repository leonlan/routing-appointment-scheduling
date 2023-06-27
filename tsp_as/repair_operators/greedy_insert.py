from copy import copy

from numpy.testing import assert_

from tsp_as.appointment.heavy_traffic import compute_schedule as compute_ht_schedule
from tsp_as.classes import Solution


def greedy_insert(solution: Solution, rng, data, cost_evaluator, **kwargs):
    """
    Insert the unassigned customers into the best place, one-by-one.
    """
    unassigned = copy(solution.unassigned)
    rng.shuffle(unassigned)

    visits = copy(solution.visits)

    while unassigned:
        customer = unassigned.pop()
        best_idx = _best_insert_idx(visits, customer, data, cost_evaluator)
        visits.insert(best_idx, customer)

    return _create_new_ht_solution(visits, data, cost_evaluator)


def _best_insert_idx(visits: list[int], customer: int, data, cost_evaluator):
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

        if _compute_new_travel_cost(new_visits, data, cost_evaluator) > best_cost:
            # If the new travel cost is already higher than the best cost, we
            # can stop the search to avoid expensive idle/wait computations.
            continue

        new = _create_new_ht_solution(new_visits, data, cost_evaluator)

        if best_cost is None or new.cost < best_cost:
            best_cost = new.cost
            best_idx = idx

    assert_(best_idx is not None)  # must always find an insertion index

    return best_idx


def _create_new_ht_solution(visits, data, cost_evaluator):
    """
    Create a new solution from the given list of client visits by computing
    a heavy traffic schedule.
    """
    ht_schedule = compute_ht_schedule(visits, data, cost_evaluator)
    return Solution(data, cost_evaluator, visits, ht_schedule)


def _compute_new_travel_cost(visits, data, cost_evaluator) -> float:
    """
    Compute the travel cost of the given visits order.
    """
    distance = data.distances[[0] + visits, visits + [0]].sum()
    return cost_evaluator.travel_weight * distance
