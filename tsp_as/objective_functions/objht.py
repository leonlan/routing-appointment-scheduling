import numpy as np


def objht(x, B, omega_b):
    n = len(x)
    obj = 0

    for i in range(0, n):
        obj += np.sqrt(B[i])

    obj = np.sqrt(2 * omega_b * (1 - omega_b)) * obj

    return obj
