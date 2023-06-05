from copy import copy

from tsp_as.appointment.heavy_traffic import compute_schedule as compute_ht_schedule
from tsp_as.appointment.true_optimal import compute_idle_wait
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
        delta_dist = _delta_dist(solution, idx, customer)
        insert_cost_dist = solution.cost_evaluator.travel_weight * delta_dist

        if insert_cost_dist > best_cost:
            # If the travel cost is already higher than the best overall cost,
            # we can stop searching.
            continue

        # Create the visits and schedule of the new solution
        new_visits = copy(solution.visits)
        new_visits.insert(idx, customer)
        new_schedule = compute_ht_schedule(
            new_visits, solution.data, solution.cost_evaluator
        )

        # Re-compute the travel, idle and waiting time of the new solution
        travel = solution.distance + delta_dist
        idle, wait = compute_idle_wait(new_visits, new_schedule, solution.data)

        # Compute the delta cost of the insertion
        delta_cost = solution.cost_evaluator.cost(new_visits, travel, idle, wait)
        delta_cost -= solution.cost_evaluator(solution)

        if best_cost is None or delta_cost < best_cost:
            best_cost = delta_cost
            best_idx = idx

    assert best_idx is not None  # Sanity check that we always find a best_idx

    solution.insert(best_idx, customer)


def _delta_dist(solution, idx, cust):
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

    return delta
