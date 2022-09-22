from .heavy_traffic import compute_schedule
from .tour2params import tour2params
from .true_optimal import compute_objective_


def heavy_traffic_optimal(tour, params):
    means, SCVs = tour2params([0] + tour, params)
    x = compute_schedule(means, SCVs, params.omega_b)
    cost = compute_objective_(x, means, SCVs, params.omega_b)
    return x, cost
