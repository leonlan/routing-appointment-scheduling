import math

import numpy as np
from scipy.linalg import inv  # matrix inversion
from scipy.linalg.blas import dgemm, dgemv  # matrix multiplication
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import expm  # matrix exponential
from scipy.stats import poisson


def phase_parameters(mean, SCV):
    """
    Returns the initial distribution gamma and the transition rate
    matrix T of the phase-fitted service times given the mean, SCV,
    and the elapsed service time u of the client in service.
    """
    if SCV < 1:  # Weighted Erlang case
        # REVIEW how is this calculated?
        K = math.floor(1 / SCV)
        # REVIEW where is the formula for this?
        p = ((K + 1) * SCV - math.sqrt((K + 1) * (1 - K * SCV))) / (SCV + 1)
        # REVIEW where is the formula for this?
        mu = (K + 1 - p) / mean

        # REVIEW where/what is gamma?
        gamma_i = np.zeros((1, K + 1))
        # REVIEW where/what is B-sf?
        B_sf = poisson.cdf(K - 1, mu) + (1 - p) * poisson.pmf(K, mu)

        # REVIEW where is this found?
        gamma_i[0, :] = [poisson.pmf(z, mu) / B_sf for z in range(K + 1)]
        gamma_i[0, K] *= 1 - p

        transition = -mu * np.eye(K + 1)
        transition += mu * np.diag(np.ones(K), k=1)  # one above diagonal
        # REVIEW why is this element multiplied by (1-p)?
        transition[K - 1, K] = (1 - p) * mu

    else:  # hyperexponential case
        p = (1 + np.sqrt((SCV - 1) / (SCV + 1))) / 2
        mu1 = 2 * p / mean  # REVIEW where can this be found?
        mu2 = 2 * (1 - p) / mean  # REVIEW where can this be found?

        B_sf = p * np.exp(-mu1) + (1 - p) * np.exp(-mu2)
        elt = p * np.exp(-mu1) / B_sf  # TODO find better name than element

        gamma_i = np.array([elt, 1 - elt])
        transition = np.diag([-mu1, -mu2])

    return gamma_i, transition


def create_Vn(gamma, T):
    """
    Creates the matrix Vn given the initial distributions `gamma` and the
    corresponding transition matrices `T`.
    """
    # initialize Vn
    n = len(T)
    d = [T[i].shape[0] for i in range(n)]
    dim_V = np.cumsum([0] + d)
    Vn = np.zeros((dim_V[n], dim_V[n]))

    # compute Vn recursively
    for i in range(1, n):
        Vn[dim_V[i - 1] : dim_V[i], dim_V[i - 1] : dim_V[i]] = T[i - 1]
        Vn[dim_V[i - 1] : dim_V[i], dim_V[i] : dim_V[i + 1]] = (
            np.matrix(-T[i - 1] @ np.ones((d[i - 1], 1))) @ gamma[i]
        )

    Vn[dim_V[n - 1] : dim_V[n], dim_V[n - 1] : dim_V[n]] = T[n - 1]

    return Vn


def compute_objective(x, gamma, Vn, Vn_inv, omega_b):
    """
    Compute the objective value of a schedule.
    """
    n = len(gamma)

    Pi = gamma[0]
    cost = omega_b * np.sum(x)
    sum_di = 0

    # cost of clients to be scheduled
    for i in range(1, n + 1):
        sum_di += gamma[i - 1].shape[1]
        exp_Vi = expm(Vn[0:sum_di, 0:sum_di] * x[i - 1])
        cost += float(
            dgemv(
                1,
                dgemm(1, Pi, Vn_inv[0:sum_di, 0:sum_di]),
                np.sum(omega_b * np.eye(sum_di) - exp_Vi, 1),
            )
        )

        if i == n:
            break

        P = dgemm(1, Pi, exp_Vi)
        Fi = 1 - np.sum(P)
        Pi = np.hstack((np.matrix(P), gamma[i] * Fi))

    return cost


def compute_objective_(x, means, SCVs, omega_b):
    """
    TODO This functions is used for HTM.
    Not sure if these args can be used for both.
    """
    n = len(means)
    gamma, T = zip(*[phase_parameters(means[i], SCVs[i]) for i in range(n)])
    Vn = create_Vn(gamma, T)
    Vn_inv = inv(Vn)

    return compute_objective(x, gamma, Vn, Vn_inv, omega_b)


def compute_schedule(means, SCVs, omega_b, tol=None):
    """
    Return the appointment times and the cost of the true optimal schedule.
    """
    n = len(means)
    gamma, T = zip(*[phase_parameters(means[i], SCVs[i]) for i in range(n)])
    Vn = create_Vn(gamma, T)
    Vn_inv = inv(Vn)

    def cost_fun(x):
        return compute_objective(x, gamma, Vn, Vn_inv, omega_b)

    x_init = 1.5 * np.ones(n)
    lin_cons = LinearConstraint(np.eye(n), 0, np.inf)
    optim = minimize(cost_fun, x_init, constraints=lin_cons, method="SLSQP", tol=tol)

    return optim.x, optim.fun
