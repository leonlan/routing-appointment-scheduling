import math

import numpy as np
from scipy.linalg import inv
from scipy.linalg.blas import dgemm
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import expm
from scipy.stats import poisson

from .tour2params import tour2params


def phase_parameters(mean, SCV):
    """
    Returns the initial distribution alpha and the transition rate
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

        # REVIEW where/what is alpha?
        alpha_i = np.zeros((1, K + 1))
        # REVIEW where/what is B-sf?
        B_sf = poisson.cdf(K - 1, mu) + (1 - p) * poisson.pmf(K, mu)

        # REVIEW where is this found?
        alpha_i[0, :] = [poisson.pmf(z, mu) / B_sf for z in range(K + 1)]
        alpha_i[0, K] *= 1 - p

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

        alpha_i = np.array([elt, 1 - elt])
        transition = np.diag([-mu1, -mu2])

    return alpha_i, transition


def create_Vn(alphas, T):
    """
    Creates the Vn matrix given the initial distributions `alphas` and the
    corresponding transition matrices `T`.

    alphas
        List of initial distribution arrays
    T
        List of transition matrices
    """
    n = len(T)
    d = [T[i].shape[0] for i in range(n)]
    dims = np.cumsum([0] + d)

    Vn = np.zeros((dims[n], dims[n]))

    for i in range(1, n):
        l = dims[i - 1]
        u = dims[i]
        k = dims[i + 1]

        Vn[l:u, l:u] = T[i - 1]
        Vn[l:u, u:k] = -T[i - 1] @ np.ones((d[i - 1], 1)) @ alphas[i]

    Vn[dims[n - 1] : dims[n], dims[n - 1] : dims[n]] = T[n - 1]

    return Vn


def compute_objective(x, alphas, Vn, omega_b):
    """
    Compute the objective value of a schedule. See Theorem (1).

    x
        The interappointment times.
    alphas
        The alpha parameters of the phase-type distribution.
    Vn
        The recursively-defined matrix $V^{(n)}$.
    omega_b
        The weight associated with idle time.
        # TODO This should become part of the params
        # TODO Check if this is idle time or waiting time.

    """
    n = len(alphas)
    omega = 0  # TODO this need to be added as parameter at some point
    dims = np.cumsum([alphas[i].size for i in range(n)])
    Vn_inv = inv(Vn)

    beta = alphas[0]
    cost = omega_b * np.sum(x)

    for i in range(n):
        d = dims[i]
        expVx = expm(Vn[:d, :d] * x[i])

        # The idle and waiting terms in the objective function can be decomposed
        # in the following three terms, reducing several matrix computations.
        term1 = dgemm(1, beta, Vn_inv[:d, :d])
        term2 = (omega_b * np.eye(d) - (1 - omega) * expVx).sum(axis=1)
        term3 = omega_b * x[i]

        cost += np.dot(term1, term2)[0] + term3

        if i == n - 1:  # stop
            break

        P = dgemm(1, beta, expVx)
        Fi = 1 - np.sum(P)
        beta = np.hstack((P, alphas[i + 1] * Fi))

    return cost


def compute_objective_(x, means, SCVs, omega_b):
    """
    TODO This functions is used for HTM. Try to refactor with ``compute_objective``.
    """
    n = len(means)
    alpha, T = zip(*[phase_parameters(means[i], SCVs[i]) for i in range(n)])
    Vn = create_Vn(alpha, T)

    return compute_objective(x, alpha, Vn, omega_b)


def compute_schedule(means, SCVs, omega_b, tol=None):
    """
    Return the appointment times and the cost of the true optimal schedule.
    """
    n = len(means)
    alpha, T = zip(*[phase_parameters(means[i], SCVs[i]) for i in range(n)])
    Vn = create_Vn(alpha, T)

    def cost_fun(x):
        return compute_objective(x, alpha, Vn, omega_b)

    x_init = 1.5 * np.ones(n)
    lin_cons = LinearConstraint(np.eye(n), 0, np.inf)
    optim = minimize(cost_fun, x_init, constraints=lin_cons, method="SLSQP", tol=tol)

    return optim.x, optim.fun


def true_optimal(tour, params):
    means, SCVs = tour2params([0] + tour, params)
    x, cost = compute_schedule(means, SCVs, params.omega_b, tol=1e-2)
    return x, cost
