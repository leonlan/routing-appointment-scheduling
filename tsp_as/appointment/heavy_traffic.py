import numpy as np

from .tour2params import tour2params


def compute_schedule(means, SCVs, omega_b):
    """
    Computes the schedule using heavy traffic approximation.

    Eq. (2) in draft.
    """
    var = SCVs * means**2  # Variance of U
    B = _compute_service_times(var)

    # Eq. (2) but without omega from travels
    coeff = (1 - omega_b) / (2 * omega_b)
    return means + np.sqrt(coeff * B)


def compute_objective(means, SCVs, omega_bar):
    """
    Computes the objective value using heavy traffic approximation.
    See (3) in draft.
    """
    var = SCVs * means**2  # Variance of U
    B = _compute_service_times(var)

    weight = np.sqrt(2 * omega_bar * (1 - omega_bar))
    return weight * np.sqrt(B).sum()


def _compute_service_times(var):
    BETA = 0.5  # TODO this should be a parameter
    n = len(var)

    beta = BETA * np.ones(n)
    betas = np.power(beta, np.arange(n))  # b^0, b^1, ..., b^{n-1}
    beta_var = betas * var  # b^0 * U_0, b^1 * U_1, ..., b^{n-1} * U_{n-1}

    # Eq (?) for S_i on page 3.
    return np.cumsum(beta_var) / np.cumsum(betas)


def heavy_traffic_pure(tour, params):
    means, SCVs = tour2params([0] + tour, params)
    x = compute_schedule(means, SCVs, params.omega_b)
    cost = compute_objective(means, SCVs, params.omega_b)

    return x, cost
