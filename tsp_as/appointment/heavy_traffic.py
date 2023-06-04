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
    to = tour  # TODO visits2tour
    travel_times = data.means[frm, to]

    idle_times = schedule - travel_times
    wait_times = S / (2 * idle_times)

    return idle_times, wait_times


def _compute_weighted_mean_variance(tour, data):
    """
    This function mimics the notation in the paper for easy checking.
    There's a much more compact way to do this, see the this commit:

    https://github.com/leonlan/tsp-as/blob/7bd9008/tsp_as/appointment/ heavy_traffic.py#L30
    """
    n = len(tour)
    BETA = 0.5  # TODO I'm not sure what this paramater is.

    # None, Var(U_1), Var(U_2), ..., Var(U_n)
    variances = [None] + get_vars(tour, data).tolist()

    # b^0, b^1, ..., b^{n}
    betas = BETA ** np.arange(n + 1)

    # S_i = \sum_{j=1}^i b^{i-j} * Var(U_j) / \sum_{j=1}^i b^{i-j}
    S = []
    for i in range(1, n + 1):  # i = 1, ..., n
        num = sum(betas[i - j] * variances[j] for j in range(1, i + 1))
        denom = sum(betas[i - j] for j in range(1, i + 1))
        S.append(num / denom)

    return S
