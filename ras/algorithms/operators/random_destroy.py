from copy import deepcopy

from numpy.random import Generator
from numpy.testing import assert_equal

from ras.classes import CostEvaluator, ProblemData, Solution


def random_destroy(
    solution: Solution,
    rng: Generator,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    max_num_destroy: int = 6,
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
    max_num_destroy
        The maximum number of clients to remove from the solution.
    """
    routes = deepcopy([route.clients for route in solution.routes])
    clients = [client for route in routes for client in route]
    num_destroy = rng.integers(1, min(max_num_destroy, len(clients)))

    for client in rng.choice(clients, num_destroy, replace=False):
        for route in routes:
            if client in route:
                route.remove(client)
                break

    destroyed = Solution.from_routes(data, cost_evaluator, routes)

    assert_equal(len(solution) - len(destroyed), num_destroy)

    return destroyed
