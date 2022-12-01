import numpy as np

from .utils import get_means_scvs


def compute_schedule(tour, params):
    """
    Computes the schedule using heavy traffic approximation.

    Eq. (2) in draft.
    """
    means, scvs = get_means_scvs(tour, params)

    var = scvs * means**2  # Variance of U
    B = _compute_service_times(var)

    # Eq. (2) but without omega from travels
    coeff = (1 - params.omega_wait - params.omega_idle) / (2 * params.omega_idle)
    return means + np.sqrt(coeff * B)


def compute_objective(tour, params):
    """
    Computes the objective value using heavy traffic approximation.
    See (3) in draft.
    """
    means, scvs = get_means_scvs(tour, params)

    var = scvs * means**2  # Variance of U
    B = _compute_service_times(var)

    weight = np.sqrt(
        2 * params.omega_idle * (1 - params.omega_wait - params.omega_idle)
    )
    return weight * np.sqrt(B).sum()


def _compute_service_times(var):
    # TODO Cache this function
    BETA = 0.5  # TODO this should be a parameter
    n = len(var)

    beta = BETA * np.ones(n)
    betas = np.power(beta, np.arange(n))  # b^0, b^1, ..., b^{n-1}
    beta_var = betas * var  # b^0 * U_0, b^1 * U_1, ..., b^{n-1} * U_{n-1}

    # Eq (?) for S_i on page 3.
    return np.cumsum(beta_var) / np.cumsum(betas)
