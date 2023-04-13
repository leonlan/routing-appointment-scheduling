from copy import deepcopy

from .solve_alns import solve_alns


def solve_tsp(seed, data, **kwargs):
    """
    Solves the TSP without appointment scheduling.
    """
    tsp_data = deepcopy(data)

    # Set omegas to zero, meaning that we purely focus on routing
    tsp_data.omega_idle = 0
    tsp_data.omega_wait = 0

    return solve_alns(seed, data=tsp_data, **kwargs)
