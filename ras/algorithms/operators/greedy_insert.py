from copy import copy

from ras.CostEvaluator import CostEvaluator
from ras.ProblemData import ProblemData
from ras.Solution import Route, Solution

from .utils import compute_unassigned


def greedy_insert(
    solution: Solution, rng, data: ProblemData, cost_evaluator: CostEvaluator, **kwargs
):
    """
    Insert the unassigned clients into the best place, one-by-one.
    """
    routes = copy(solution.routes)
    unassigned = compute_unassigned(data, solution)
    rng.shuffle(unassigned)

    while unassigned:
        client = unassigned.pop()
        route_idx, idx = _best_insert_idx(data, cost_evaluator, routes, client)

        clients = copy(routes[route_idx].clients)
        clients.insert(idx, client)
        routes[route_idx] = Route.from_clients(data, cost_evaluator, clients)

    return Solution(routes)


def _best_insert_idx(
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    routes: list[Route],
    client: int,
) -> tuple[int, int]:
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
    tuple[int, int]
        The index of the best route and the index to insert the client.
    """
    best_route_idx = None
    best_idx = None
    best_cost = float("inf")

    for route_idx, route in enumerate(routes):
        for idx in range(len(route) + 1):
            new_visits = copy(route.clients)
            new_visits.insert(idx, client)

            if _compute_new_travel_cost(data, cost_evaluator, new_visits) > best_cost:
                # If the new travel cost is already higher than the best cost, we
                # can stop the search to avoid expensive idle/wait computations.
                continue

            new_cost = Route.from_clients(data, cost_evaluator, new_visits).cost
            delta_cost = new_cost - route.cost

            if best_cost is None or delta_cost < best_cost:
                best_route_idx = route_idx
                best_idx = idx
                best_cost = delta_cost

    # Must always find an insertion index.
    assert best_route_idx is not None and best_idx is not None

    return best_route_idx, best_idx


def _compute_new_travel_cost(
    data: ProblemData, cost_evaluator: CostEvaluator, visits: list[int]
) -> float:
    """
    Compute the travel cost of the given visits order.
    """
    distance = data.distances[[0] + visits, visits + [0]].sum()
    return cost_evaluator.travel_weight * distance
