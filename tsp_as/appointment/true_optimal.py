import numpy as np
from scipy.linalg import inv
from scipy.linalg.blas import dgemm
from scipy.optimize import minimize
from scipy.sparse.linalg import expm


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
    omega_b = params.omega_b
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
    Compute the objective function assuming that the schedule is given. This
    is used for the mixed heavy traffic and true optimal strategy.
    """
    fr = [0] + tour
    to = tour + [0]

    alpha = tuple(params.alphas[fr, to])
    T = tuple(params.transitions[fr, to])
    Vn = create_Vn(alpha, T)

    return compute_objective(x, alpha, Vn, params)


def compute_optimal_schedule(tour, params, **kwargs):
    """
    Computes the optimal schedule of the tour by minimizing the true optimal
    objective function.
    """
    fr = [0] + tour
    to = tour + [0]

    alpha = tuple(params.alphas[fr, to])
    T = tuple(params.transitions[fr, to])
    Vn = create_Vn(alpha, T)

    def cost_fun(x):
        return compute_objective(x, alpha, Vn, params)

    x_init = 1.5 * np.ones(len(fr))

    optim = minimize(
        cost_fun,
        x_init,
        method="SLSQP",
        tol=kwargs.get("tol", 0.01),
        bounds=[(0, None) for _ in range(x_init.size)],
    )

    return optim.x, optim.fun
