import numpy as np


def compute_schedule(tour, params):
    """
    Computes the schedule using heavy traffic approximation.

    Eq. (2) in draft.
    """
    fr = [0] + tour
    to = tour + [0]

    means = params.means[fr, to]
    SCVs = params.scvs[fr, to]

    var = SCVs * means**2  # Variance of U
    B = _compute_service_times(var)

    # Eq. (2) but without omega from travels
    coeff = (1 - params.omega_b) / (2 * params.omega_b)
    return means + np.sqrt(coeff * B)


def compute_objective(tour, params):
    """
    Computes the objective value using heavy traffic approximation.
    See (3) in draft.
    """
    fr = [0] + tour
    to = tour + [0]

    means = params.means[fr, to]
    SCVs = params.scvs[fr, to]

    var = SCVs * means**2  # Variance of U
    B = _compute_service_times(var)

    weight = np.sqrt(2 * params.omega_b * (1 - params.omega_b))
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
