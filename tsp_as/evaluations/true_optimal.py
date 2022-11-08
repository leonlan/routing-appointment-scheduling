import math

import numpy as np
from scipy.linalg import inv  # matrix inversion
from scipy.linalg.blas import dgemm, dgemv  # matrix multiplication
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import expm  # matrix exponential
from scipy.stats import poisson

from .tour2params import tour2params


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
    # Initialize parameters
    n = len(T)
    d = [T[idx].shape[0] for idx in range(n)]
    dim_V = np.cumsum(d)  # dimensions to find indices in Vn
    dim_Vn = dim_V[-1]  # dimension of the final matrix Vn
    Vn = np.zeros((dim_Vn, dim_Vn))

    # Compute Vn recursively
    for i in range(n):
        l = dim_V[i - 1] if i > 0 else 0
        u = dim_V[i]
        t = T[i - 1]

        Vn[l:u, l:u] = t

        if i != n - 1:
            k = dim_V[i + 1]
            Vn[l:u, u:k] = -t @ np.ones((d[i - 1], 1)) @ gamma[i]

    return Vn


def compute_objective(x, gamma, Vn, Vn_inv, omega_b):
    """
    Compute the objective value of a schedule.

    x: np.array
        the interappointment times
    gamma: ...

    Theorem (1).
    """
    n = len(gamma)
    Pi = gamma[0]

    cost = omega_b * np.sum(x)
    dim_csum = np.cumsum([gamma[i].shape[1] for i in range(n)])  # REVIEW why shape 1?

    # cost of clients to be scheduled
    for i in range(n):
        d = dim_csum[i]
        exp_Vi = expm(Vn[:d, :d] * x[i])

        expr = dgemv(  # dgemv returns np.array
            1,
            dgemm(1, Pi, Vn_inv[:d, :d]),
            np.sum(omega_b * np.eye(d) - exp_Vi, 1),
        )[0]
        cost += expr
        # breakpoint()

        if i == n - 1:  # stop
            break

        P = dgemm(1, Pi, exp_Vi)
        Fi = 1 - np.sum(P)
        Pi = np.concatenate((P, gamma[i + 1] * Fi), axis=1)

    return cost


def compute_objective_(x, means, SCVs, omega_b):
    """
    TODO This functions is used for HTM. Try to refactor with ``compute_objective``.
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


def true_optimal(tour, params):
    means, SCVs = tour2params([0] + tour, params)
    x, cost = compute_schedule(means, SCVs, params.omega_b, tol=1e-2)
    return x, cost
