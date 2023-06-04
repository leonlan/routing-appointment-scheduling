import numpy as np

from .utils import get_leg_data, get_vars


def compute_schedule(tour, data, cost_evaluator):
    """
    Computes the schedule using heavy traffic approximation.

    Eq. (2) in draft.
    """
    means, _, _ = get_leg_data(tour, data)
    S = _compute_weighted_mean_variance(tour, data)

    wait_weights = cost_evaluator.wait_weights[tour]
    idle_weight = cost_evaluator.idle_weight

    return means + np.sqrt((wait_weights * S) / (2 * idle_weight))


def compute_objective(tour, schedule, data) -> tuple[list[float], list[float]]:
    """
    Computes the objective value using heavy traffic approximation.
    See (3) in draft.
    """
    S = _compute_weighted_mean_variance(tour, data)
    frm = [0] + tour[:-1]
    to = tour
    travel_times = data.means[frm, to]

    idle_times = schedule - travel_times
    wait_times = S / (2 * idle_times)

    return idle_times, wait_times


def _compute_weighted_mean_variance(tour, data):
    variances = get_vars(tour, data)
    n = len(variances)
    BETA = 0.5  # TODO this should be a parameter

    beta = BETA * np.ones(n)
    betas = np.power(beta, np.arange(n))  # b^0, b^1, ..., b^{n-1}
    beta_var = betas * variances  # b^0 * U_0, b^1 * U_1, ..., b^{n-1} * U_{n-1}

    return np.cumsum(beta_var) / np.cumsum(betas)
