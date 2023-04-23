import numpy as np

from .utils import get_leg_data


def compute_schedule(tour, data):
    """
    Computes the schedule using heavy traffic approximation.

    Eq. (2) in draft.
    """
    means, _, variances = get_leg_data(tour, data)
    B = _compute_service_times(variances)
    coeff = (1 - data.omega_travel - data.omega_idle) / (2 * data.omega_idle)
    return means + np.sqrt(coeff * B)


def compute_objective(tour, data):
    """
    Computes the objective value using heavy traffic approximation.
    See (3) in draft.
    """
    variances = get_vars(tour, data)
    B = _compute_service_times(variances)

    coeff = np.sqrt(2 * data.omega_idle * (1 - data.omega_travel - data.omega_idle))
    return coeff * np.sqrt(B).sum()


def _compute_service_times(var):
    BETA = 0.5  # TODO this should be a parameter
    n = len(var)

    beta = BETA * np.ones(n)
    betas = np.power(beta, np.arange(n))  # b^0, b^1, ..., b^{n-1}
    beta_var = betas * var  # b^0 * U_0, b^1 * U_1, ..., b^{n-1} * U_{n-1}

    return np.cumsum(beta_var) / np.cumsum(betas)


def get_vars(tour, data):
    frm = [0] + tour[:-1]
    to = tour

    return data.vars[frm, to]
