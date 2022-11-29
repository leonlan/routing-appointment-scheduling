import math

import numpy as np
from scipy.linalg import inv
from scipy.linalg.blas import dgemm
from scipy.optimize import minimize
from scipy.sparse.linalg import expm
from scipy.stats import poisson


def phase_parameters(mean, SCV):
    """
    Returns the initial distribution alpha and the transition rate
    matrix T of the phase-fitted service times given the mean, SCV,
    and the elapsed service time u of the client in service.

    # TODO These phase parameters can be computed in params as well?
    """
    if SCV < 1:  # Weighted Erlang case
        K = math.floor(1 / SCV)
        prob = ((K + 1) * SCV - math.sqrt((K + 1) * (1 - K * SCV))) / (SCV + 1)
        mu = (K + 1 - prob) / mean

        alpha = np.zeros((1, K + 1))
        B_sf = poisson.cdf(K - 1, mu) + (1 - prob) * poisson.pmf(K, mu)

        alpha[0, :] = [poisson.pmf(z, mu) / B_sf for z in range(K + 1)]
        alpha[0, K] *= 1 - prob

        transition = -mu * np.eye(K + 1)
        transition += mu * np.diag(np.ones(K), k=1)  # one above diagonal
        transition[K - 1, K] = (1 - prob) * mu

    else:  # Hyperexponential case
        prob = (1 + np.sqrt((SCV - 1) / (SCV + 1))) / 2
        mu1 = 2 * prob / mean
        mu2 = 2 * (1 - prob) / mean

        B_sf = prob * np.exp(-mu1) + (1 - prob) * np.exp(-mu2)
        term = prob * np.exp(-mu1) / B_sf

        alpha = np.array([term, 1 - term])
        transition = np.diag([-mu1, -mu2])

    return alpha, transition


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


def compute_objective_given_schedule(tour, x, params):
    """
    TODO This functions is used for HTM. Try to refactor with ``compute_objective``.
    """
    fr = [0] + tour
    to = tour + [0]

    means = params.means[fr, to]
    SCVs = params.scvs[fr, to]

    n = len(means)
    alpha, T = zip(*[phase_parameters(means[i], SCVs[i]) for i in range(n)])
    Vn = create_Vn(alpha, T)

    return compute_objective(x, alpha, Vn, params.omega_b)


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

    optim = minimize(
        cost_fun,
        x_init,
        method="SLSQP",
        tol=tol,
        bounds=[(0, None) for _ in range(x_init.size)],
    )

    return optim.x, optim.fun


def compute_optimal_schedule(tour, params):
    fr = [0] + tour
    to = tour + [0]

    means = params.means[fr, to]
    SCVs = params.scvs[fr, to]

    x, cost = compute_schedule(means, SCVs, params.omega_b, tol=1e-2)
    return x, cost
