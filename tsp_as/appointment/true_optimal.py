import numpy as np
from scipy.linalg import inv
from scipy.linalg.blas import dgemm
from scipy.optimize import minimize
from scipy.sparse.linalg import expm

from tsp_as.appointment.utils import get_alphas_transitions

from .heavy_traffic import compute_schedule as ht_compute_schedule
from .utils import get_alphas_transitions


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


def compute_objective(x, alphas, Vn, params):
    """
    Compute the objective value of a schedule. See Theorem (1).

    x
        The interappointment times.
    alphas
        The alpha parameters of the phase-type distribution.
    Vn
        The recursively-defined matrix $V^{(n)}$.
    params
        The parameters of the problem.
    """
    n = len(alphas)
    omega_idle = params.omega_idle
    omega_travel = params.omega_travel
    dims = np.cumsum([alphas[i].size for i in range(n)])
    Vn_inv = inv(Vn)

    beta = alphas[0]
    cost = omega_idle * np.sum(x)

    for i in range(n):
        d = dims[i]
        expVx = expm(Vn[:d, :d] * x[i])

        # The idle and waiting terms in the objective function can be decomposed
        # in the following three terms, reducing several matrix computations.
        term1 = dgemm(1, beta, Vn_inv[:d, :d])
        term2 = (omega_idle * np.eye(d) - (1 - omega_travel) * expVx).sum(axis=1)
        term3 = omega_idle * x[i]

        cost += np.dot(term1, term2)[0] + term3

        if i == n - 1:  # stop
            break

        P = dgemm(1, beta, expVx)
        Fi = 1 - np.sum(P)
        beta = np.hstack((P, alphas[i + 1] * Fi))

    return cost


def compute_objective_given_schedule(tour, x, params):
    """
    Compute the objective function assuming that the schedule is given. This
    is used for the mixed heavy traffic and true optimal strategy.
    """
    alpha, T = get_alphas_transitions(tour, params)
    Vn = create_Vn(alpha, T)

    return compute_objective(x, alpha, Vn, params)


def compute_optimal_schedule(tour, params, warmstart=True, **kwargs):
    """
    Computes the optimal schedule of the tour by minimizing the true optimal
    objective function.
    """
    alpha, T = get_alphas_transitions(tour, params)
    Vn = create_Vn(alpha, T)

    def cost_fun(x):
        return compute_objective(x, alpha, Vn, params)

    if warmstart:
        x_init = ht_compute_schedule(tour, params)
    else:
        # Use means of travel time and service as initial value
        x_init = params.distances[[0] + tour, tour + [0]] + params.service[[0] + tour]

    optim = minimize(
        cost_fun,
        x_init,
        method=kwargs.get("method", "trust-constr"),
        tol=kwargs.get("tol", 0.01),
        bounds=[(0, None) for _ in range(x_init.size)],
        options={"disp": True},
    )

    return optim.x, optim.fun
