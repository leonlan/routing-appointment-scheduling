import numpy as np
from scipy.linalg import expm, inv
from scipy.linalg.blas import dgemm
from scipy.optimize import minimize

from .heavy_traffic import compute_schedule as ht_compute_schedule
from .utils import get_alphas_transitions


def _compute_idle_wait_per_client(x, alpha, Vn, *, lag=False):
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
    lag
        Whether to return the idle and wait times for the last client only,
        which is used for the lag-based objective function.
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
            if lag:  # If used as subprocedure in the lag-based obj function
                return idle.item(), wait.item()
            break

        P = dgemm(1, beta, expVx)
        Fi = 1 - np.sum(P)
        beta = np.hstack((P, alpha[i + 1] * Fi))

    return idle_times, wait_times


def compute_idle_wait(tour, schedule, data):
    """
    Compute the idle and wait times for a solution (tour and schedule).

    Parameters
    ----------
    tour
        The tour.
    schedule
        The interappointment times.
    data
        The problem data.
    """
    idle_times, wait_times = compute_idle_wait_per_client(tour, schedule, data)
    return sum(idle_times), sum(wait_times)


def compute_idle_wait_per_client(tour, schedule, data):
    """
    Compute the idle and wait times per client for a solution (tour and schedule).

    Parameters
    ----------
    tour
        The tour.
    schedule
        The interappointment times.
    data
        The problem data.
    """
    alpha, Vn = _get_alphas_and_Vn(tour, data)

    idle_times, wait_times = _compute_idle_wait_per_client(schedule, alpha, Vn)
    return idle_times, wait_times


def compute_schedule_and_idle_wait(tour, data, warmstart=True, **kwargs):
    """
    Compute the optimal schedule and the corresponding idle and wait times.

    Parameters
    ----------
    tour
        The tour.
    data
        The problem data.
    warmstart
        Whether to use the heavy traffic schedule as initial value.
    """
    alpha, Vn = _get_alphas_and_Vn(tour, data)

    def cost_fun(x):
        idle, wait = _compute_idle_wait_per_client(x, alpha, Vn)
        return sum(idle) + sum(wait)

    if warmstart:
        x_init = ht_compute_schedule(tour, data)
    else:  # Use means of travel time and service as initial value
        x_init = data.distances[[0] + tour, tour + [0]] + data.service[[0] + tour]

    optim = minimize(
        cost_fun,
        x_init,
        method=kwargs.get("method", "trust-constr"),
        tol=kwargs.get("tol", 0.01),
        bounds=[(0, None) for _ in range(x_init.size)],
        options={"disp": True},
    )

    return optim.x, *compute_idle_wait(tour, optim.x, data)


def _get_alphas_and_Vn(tour, data):
    alpha, T = get_alphas_transitions(tour, data)
    Vn = _create_Vn(alpha, T)
    return alpha, Vn


def _create_Vn(alphas, T):
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
