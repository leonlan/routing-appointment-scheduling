import numpy as np
from scipy.linalg import inv
from scipy.linalg.blas import dgemm
from scipy.sparse.linalg import expm

import tsp_as.appointment.true_optimal as to
from tsp_as.appointment.utils import get_alphas_transitions

from .utils import get_alphas_transitions


def compute_objective_given_schedule(tour, x, params):
    """
    Compute the objective value of a schedule using the lag-based objective
    function.

    Parameters
    ----------
    tour
        The client visiting tour.
    x
        The interappointment times.
    params
        The problem parameters.
    """
    L = params.lag

    alpha, T = get_alphas_transitions(tour, params)

    n = len(alpha)
    cost = params.omega_idle * np.sum(x)

    # We take a "slice" of the x, alphas to make a new Vn matrix. We pass
    # these as input to the true optimal objective function each time.
    for i in range(n):
        l = max(0, i - L)  # lower index
        u = i + 1  # upper index

        cost += to.compute_objective(
            x[l:u],
            alpha[l:u],
            to.create_Vn(alpha[l:u], T[l:u]),
            params,
            lag=True,
        )

    return cost
