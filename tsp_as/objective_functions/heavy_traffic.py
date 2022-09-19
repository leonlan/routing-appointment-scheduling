import numpy as np
from _shared import cost, create_Vn, phase_parameters
from objht import objht
from scipy.linalg import inv  # matrix inversion
from scipy.linalg.blas import dgemm, dgemv  # matrix multiplication
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import expm  # matrix exponential
from scipy.stats import poisson


def Transient_IA(means, SCVs, omega_b, tol=None):
    """
    Computes the optimal schedule.
    wis = waiting in system. # N = n + wis
    """

    n = len(means)
    gamma, T = zip(*[phase_parameters(means[i], SCVs[i]) for i in range(n)])
    Vn = create_Vn(gamma, T)
    Vn_inv = inv(Vn)

    # assigning new variables for heavy traffic computations (equation number 2 in the writeup)
    v = [None] * n
    x = [None] * n
    B = [None] * n
    Nu = [None] * n
    De = [None] * n

    nu = 0
    de = 0
    for i in range(0, n):
        v[i] = SCVs[i] * pow(means[i], 2)

    al = 0.5

    for i in range(1, n + 1):
        for j in range(0, i):
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


n = 10

omega_b = 0.8

means = [0.5] * n
SCVs = [0.5] * n
