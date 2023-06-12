import numpy as np


def compute_schedule(visits, data, cost_evaluator) -> list[float]:
    """
    Computes the schedule using heavy traffic approximation.
    """
    means = _get_arc_means(visits, data)
    S = _compute_weighted_mean_variance(visits, data)

    wait_weights = cost_evaluator.wait_weights[visits]
    idle_weight = cost_evaluator.idle_weight

    return means + np.sqrt((wait_weights * S) / (2 * idle_weight))


def compute_idle_wait(visits, schedule, data) -> tuple[list[float], list[float]]:
    """
    Computes the objective value using heavy traffic approximation.
    """
    means = _get_arc_means(visits, data)
    S = _compute_weighted_mean_variance(visits, data)

    idle_times = schedule - means
    wait_times = S / (2 * idle_times)

    return idle_times, wait_times


def _compute_weighted_mean_variance(visits, data):
    """
    This function mimics the notation in the paper for easy checking.
    There's a much more compact way to do this, see the this commit:

    https://github.com/leonlan/tsp-as/blob/7bd9008/tsp_as/appointment/heavy_traffic.py#L30
    """
    n = len(visits)
    BETA = 0.5  # TODO I'm not sure what this paramater is.

    # None, Var(U_1), Var(U_2), ..., Var(U_n)
    variances = [None] + _get_arc_vars(visits, data).tolist()

    # b^0, b^1, ..., b^{n}
    betas = BETA ** np.arange(n + 1)

    # S_i = \sum_{j=1}^i b^{i-j} * Var(U_j) / \sum_{j=1}^i b^{i-j}
    S = []
    for i in range(1, n + 1):  # i = 1, ..., n
        num = sum(betas[i - j] * variances[j] for j in range(1, i + 1))
        denom = sum(betas[i - j] for j in range(1, i + 1))
        S.append(num / denom)

    return S


def _get_arc_means(visits, data):
    return data.arcs_mean[_visits2arcs(visits)]


def _get_arc_vars(visits, data):
    return data.arcs_var[_visits2arcs(visits)]


def _visits2arcs(visits):
    """
    Returns the from and to indices for the given arcs to the client visits.
    """
    arcs = [0] + visits
    return arcs[:-1], arcs[1:]
