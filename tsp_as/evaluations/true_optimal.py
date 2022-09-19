import numpy as np
from _shared import cost, create_Vn, phase_parameters
from scipy.linalg import inv
from scipy.optimize import LinearConstraint, minimize


def compute_schedule(means, SCVs, omega_b):
    """
    Return the appointment times and the cost of the true optimal schedule.
    """
    n = len(means)
    gamma, T = zip(*[phase_parameters(means[i], SCVs[i]) for i in range(n)])
    Vn = create_Vn(gamma, T)
    Vn_inv = inv(Vn)

    def cost_fun(x):
        return cost(x, gamma, Vn, Vn_inv, omega_b)

    x_init = 1.5 * np.ones(n)
    lin_cons = LinearConstraint(np.eye(n), 0, np.inf)
    optim = minimize(cost_fun, x_init, constraints=lin_cons, method="SLSQP")

    return optim.x, optim.fun
