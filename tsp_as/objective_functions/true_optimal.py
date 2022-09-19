import numpy as np
from _shared import create_Vn, phase_parameters, cost
from scipy.linalg import inv  # matrix inversion
from scipy.linalg.blas import dgemm, dgemv  # matrix multiplication
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import expm  # matrix exponential
from scipy.stats import poisson


def Transient_IA(means, SCVs, omega, tol=None):
    """
    Computes the optimal schedule.
    wis = waiting in system. # N = n + wis
    """

    n = len(means)
    gamma, T = zip(*[phase_parameters(means[i], SCVs[i]) for i in range(n)])
    Vn = create_Vn(gamma, T)
    Vn_inv = inv(Vn)

    # minimization
    x0 = np.array([1.5] + [1.5] * (n - 1))  # initial guess, of length n
    cost_fun = lambda x: cost(x, gamma, Vn, Vn_inv, omega_b)
    lin_cons = LinearConstraint(np.eye(n), 0, np.inf)
    optim = minimize(cost_fun, x0, constraints=lin_cons, method="SLSQP", tol=tol)

    return optim.x, optim.fun


n = 10
omega_b = 0.8

means = [0.5] * n
SCVs = [0.5] * n
# means = [0.5] * 10 + [1.5] * 10 + [0.5] * 10
# print(means)
# SCVs = [0.5] * 10 + [1.2] * 10 + [0.5] * 10

Transient_IA(means, SCVs, omega_b)
