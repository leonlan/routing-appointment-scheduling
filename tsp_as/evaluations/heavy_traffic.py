import numpy as np


def compute_schedule(means, SCVs, omega_b):
    """
    Computes the optimal schedule, i.e., appointment times, using heavy traffic
    approximation.

    Eq. (2) in draft.
    """
    n = len(means)

    # TODO what is v?
    v = np.array([SCVs[i] * pow(means[i], 2) for i in range(n)])

    # assigning new variables for heavy traffic computations (2)
    B = np.zeros(n)
    nu, de, al = 0, 0, 0.5

    for i in range(1, n + 1):
        for j in range(i):
            nu += v[j] * pow(al, i - j)
            de += pow(al, i - j)

        B[i - 1] = nu / de

    # Eq. (2)
    x = [means[i] + np.sqrt((1 - omega_b) * B[i] / (2 * omega_b)) for i in range(n)]

    return x, B  # TODO remove returning B


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
