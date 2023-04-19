import numpy as np

from .utils import get_means_scvs


def compute_schedule(tour, data):
    """
    Computes the schedule using heavy traffic approximation.

    Eq. (2) in draft.
    """
    means, _ = get_means_scvs(tour, data)
    vars = get_vars(tour, data)

    B = _compute_service_times(vars)

    # TODO When `coeff` is zero, then this function doesn't work. Check with
    # group what's going on here.
    # Eq. (2) but without omega from travels
    coeff = (1 - data.omega_travel - data.omega_idle) / (2 * data.omega_idle)
    return means + np.sqrt(coeff * B)


def compute_objective(tour, data):
    """
    Computes the objective value using heavy traffic approximation.
    See (3) in draft.
    """
    vars = get_vars(tour, data)

    B = _compute_service_times(vars)

    weight = np.sqrt(2 * data.omega_idle * (1 - data.omega_travel - data.omega_idle))
    return weight * np.sqrt(B).sum()


def _compute_service_times(var):
    BETA = 0.5  # TODO this should be a parameter
    n = len(var)

    beta = BETA * np.ones(n)
    betas = np.power(beta, np.arange(n))  # b^0, b^1, ..., b^{n-1}
    beta_var = betas * var  # b^0 * U_0, b^1 * U_1, ..., b^{n-1} * U_{n-1}

    # Eq (?) for S_i on page 3.
    return np.cumsum(beta_var) / np.cumsum(betas)


def get_vars(tour, data):
    fr = [0] + tour
    to = tour + [0]

    return data.vars[fr, to]
