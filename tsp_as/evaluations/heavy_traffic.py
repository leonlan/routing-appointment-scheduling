import numpy as np


def compute_schedule(means, SCVs, omega_b):
    """
    Computes the optimal schedule, i.e., appointment times, using heavy traffic
    approximation.

    Eq. (2) in draft.
    """
    # TODO Remove this when means and SCVs are np array by default
    means = np.array(means)
    SCVs = np.array(SCVs)

    var = SCVs * means**2  # Variance of U
    B = _compute_service_times(var)

    # Eq. (2) but without omega from travels
    coeff = (1 - omega_b) / (2 * omega_b)
    return means + np.sqrt(coeff * B), B  # TODO remove service times


def compute_objective(x, B, omega_bar):
    """
    Computes the objective value using heavy traffic approximation.
    See (3) in draft.

    NOTE
    - B == S (service times)
    - Why is the regular omega not included?

    TODO
    - Add travel times
    """
    sum_sqrt_s = sum([np.sqrt(B[i]) for i in range(len(x))])
    coeff = np.sqrt(2 * omega_bar * (1 - omega_bar))
    return coeff * sum_sqrt_s


def _compute_service_times(var):
    BETA = 0.5  # TODO this should be a parameter
    n = len(var)

    beta = BETA * np.ones(n)
    betas = np.power(beta, np.arange(n))  # b^0, b^1, ..., b^{n-1}
    beta_var = betas * var  # b^0 * U_0, b^1 * U_1, ..., b^{n-1} * U_{n-1}

    # Eq (?) for S_i on page 3.
    return np.cumsum(beta_var) / np.cumsum(betas)
