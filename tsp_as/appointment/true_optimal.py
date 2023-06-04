import numpy as np
from scipy.linalg import expm, inv
from scipy.linalg.blas import dgemm
from scipy.optimize import minimize

from .heavy_traffic import compute_schedule as ht_compute_schedule


def _compute_idle_wait_per_client(x, alpha, Vn):
    """
    Computes the objective value of a schedule. See Theorem (1).

    Parameters
    ----------
    x
        The interappointment times.
    alphas
        The alpha parameters of the phase-type distribution.
    Vn
        The recursively-defined matrix $V^{(n)}$.
    data
        The problem data.
    """
    n = len(alpha)
    dims = np.cumsum([alpha[i].size for i in range(n)])
    Vn_inv = inv(Vn)  # slicing with j-th dimension gives V^(j)^{-1}
    beta = alpha[0]  # alpha_(j)'s in the paper

    idle_times = []
    wait_times = []

    for i in range(n):
        d = dims[i]  # current dimension
        Vx = Vn[:d, :d] * x[i]
        expVx = expm(Vx)
        betaVinv = dgemm(1, beta, Vn_inv[:d, :d])

        idle = -betaVinv @ dgemm(1, expVx, np.ones((d, 1)))
        wait = x[i] + betaVinv @ np.ones((d, 1)) + idle

        idle_times.append(idle.item())
        wait_times.append(wait.item())

        if i == n - 1:  # stop
            break

        P = dgemm(1, beta, expVx)
        Fi = 1 - np.sum(P)
        beta = np.hstack((P, alpha[i + 1] * Fi))

    return idle_times, wait_times


def compute_idle_wait(visits, schedule, data) -> tuple[list[float], list[float]]:
    """
    Compute the idle and wait times for a solution (visits and schedule).
    Wrapper for `_compute_idle_wait_per_client`.

    Parameters
    ----------
    visits
        The visits.
    schedule
        The interappointment times.
    data
        The problem data.

    Returns
    -------
    idle_times
        The idle times per client.
    wait_times
        The wait times per client.
    """
    return compute_idle_wait_per_client(visits, schedule, data)


def compute_idle_wait_per_client(visits, schedule, data):
    """
    Compute the idle and wait times per client for a solution (visits and schedule).

    Parameters
    ----------
    visits
        The visits.
    schedule
        The interappointment times.
    data
        The problem data.
    """
    alpha, Vn = _get_alphas_and_Vn(visits, data)

    idle_times, wait_times = _compute_idle_wait_per_client(schedule, alpha, Vn)
    return idle_times, wait_times


def compute_schedule_and_idle_wait(visits, data, cost_evaluator, **kwargs):
    """
    Compute the optimal schedule and the corresponding idle and wait times.

    Parameters
    ----------
    visits
        The visits.
    data
        The problem data.
    cost_evaluator
        The cost evaluator.
    """
    alpha, Vn = _get_alphas_and_Vn(visits, data)

    def cost_fun(x):
        idle_weight = cost_evaluator.idle_weight
        wait_weights = cost_evaluator.wait_weights[visits]

        idle, wait = _compute_idle_wait_per_client(x, alpha, Vn)
        return idle_weight * sum(idle) + np.dot(wait_weights, wait)

    # Use heavy traffic solution as initial guess
    x_init = ht_compute_schedule(visits, data, cost_evaluator)

    optim = minimize(
        cost_fun,
        x_init,
        method=kwargs.get("method", "trust-constr"),
        tol=kwargs.get("tol", 0.01),
        bounds=[(0, None) for _ in range(x_init.size)],
    )

    return optim.x, *compute_idle_wait(visits, optim.x, data)


def _create_Vn(alphas, T):
    """
    Creates the Vn matrix given the initial distributions `alphas` and the
    corresponding transition matrices `T`.

    Parameters
    ----------
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
        l = dims[i - 1]  # noqa
        u = dims[i]
        k = dims[i + 1]

        Vn[l:u, l:u] = T[i - 1]
        Vn[l:u, u:k] = -T[i - 1] @ np.ones((d[i - 1], 1)) @ alphas[i]

    Vn[dims[n - 1] : dims[n], dims[n - 1] : dims[n]] = T[n - 1]

    return Vn


def _get_alphas_and_Vn(visits, data):
    alpha, T = _get_alphas_transitions(visits, data)
    Vn = _create_Vn(alpha, T)
    return alpha, Vn


def _get_alphas_transitions(visits, data):
    arcs = [0] + visits[:-1], visits
    return tuple(data.alphas[arcs]), tuple(data.transitions[arcs])
