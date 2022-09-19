import numpy as np
from objht import objht


def Transient_IA(means, SCVs, omega_b):
    """
    Computes the optimal schedule, i.e., appointment times.

    Notes:
    - wis = waiting in system.
    """

    n = len(means)

    # TODO what is v?
    v = [SCVs[i] * pow(means[i], 2) for i in range(n)]

    # assigning new variables for heavy traffic computations
    # (equation number 2 in the writeup)
    x = np.zeros(n)
    B = np.zeros(n)
    Nu = np.zeros(n)
    De = np.zeros(n)

    nu = 0
    de = 0
    al = 0.5

    for i in range(1, n + 1):
        for j in range(i):
            nu += v[j] * pow(al, i - j)
            de += pow(al, i - j)

        Nu[i - 1] = nu
        De[i - 1] = de
        B[i - 1] = nu / de  # S(i) for heavy traffic in code

    for i in range(0, n):
        x[i] = means[i] + np.sqrt(((1 - omega_b) * B[i]) / (2 * omega_b))

    # minimization
    cost_fun_ht = objht(x, B, omega_b)  # heavy traffic loss function

    return x, cost_fun_ht
