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
    best_idx = None
    best_cost = float("inf")

    for idx in range(len(solution.visits) + 1):
        if _insert_cost_travel(solution, idx, customer) > best_cost:
            # If the travel cost is already higher than the best overall cost,
            # we can stop searching.
            continue

        cost = _insert_cost(solution, idx, customer)

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_idx = idx

    assert best_idx is not None  # Sanity check that we always find a best_idx

    solution.insert(best_idx, customer)


def _insert_cost_travel(solution, idx, cust):
    if len(solution.visits) == 0:
        pred, succ = 0, 0
    elif idx == 0:
        pred, succ = 0, solution.visits[idx]
    elif idx == len(solution.visits):
        pred, succ = solution.visits[idx - 1], 0
    else:
        pred, succ = solution.visits[idx - 1], solution.visits[idx]

    delta = solution.data.distances[pred, cust] + solution.data.distances[cust, succ]
    delta -= solution.data.distances[pred, succ]

    return solution.cost_evaluator.travel_weight * delta


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
