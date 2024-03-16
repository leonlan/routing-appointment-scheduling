from copy import copy, deepcopy

from ras.classes import CostEvaluator, ProblemData, Solution

from .utils import compute_unassigned


def greedy_insert(
    solution: Solution, rng, data: ProblemData, cost_evaluator: CostEvaluator, **kwargs
):
    """
    Insert the unassigned clients into the best place, one-by-one.
    """
    routes = deepcopy(solution.routes)
    unassigned = compute_unassigned(data, solution)
    rng.shuffle(unassigned)

    while unassigned:
        client = unassigned.pop()
        route, idx = _best_insert_idx(data, cost_evaluator, routes, client)
        route.insert(idx, client)

    return Solution.from_routes(data, cost_evaluator, routes)


def _best_insert_idx(
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    routes: list[list[int]],
    client: int,
) -> tuple[list[int], int]:
    """
    Finds the best route and insertion index for the client in the current visits order.

    We do this by trying all possible insertion points and selecting the one
    that leads to the lowest delta cost.

    Parameters
    ----------
    data
        Problem data.
    cost_evaluator
        Cost evaluator.
    routes
        The current routes.
    client
        The client to insert.

    Returns
    -------
    list[int]
        The best route.
    int
        The best insertion index.
    """
    best_route = None
    best_idx = None
    best_cost = float("inf")

    for route in routes:
        old_cost = Solution.from_routes(data, cost_evaluator, [route]).cost

        for idx in range(len(route) + 1):
            new_visits = copy(route)
            new_visits.insert(idx, client)

            if _compute_new_travel_cost(data, cost_evaluator, new_visits) > best_cost:
                # If the new travel cost is already higher than the best cost, we
                # can stop the search to avoid expensive idle/wait computations.
                continue

            new_cost = Solution.from_routes(data, cost_evaluator, [new_visits]).cost
            delta_cost = new_cost - old_cost

            if best_cost is None or delta_cost < best_cost:
                best_route = route
                best_idx = idx
                best_cost = delta_cost

    # Must always find an insertion index.
    assert best_route is not None and best_idx is not None

    return best_route, best_idx


def _compute_new_travel_cost(
    data: ProblemData, cost_evaluator: CostEvaluator, visits: list[int]
) -> float:
    """
    Compute the travel cost of the given visits order.
    """
    distance = data.distances[[0] + visits, visits + [0]].sum()
    return cost_evaluator.travel_weight * distance
