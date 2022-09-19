import numpy as np
from _shared import create_Vn, phase_parameters
from scipy.linalg import inv  # matrix inversion
from scipy.linalg.blas import dgemm, dgemv  # matrix multiplication
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import expm  # matrix exponential
from scipy.stats import poisson


def cost(x, gamma, Vn, Vn_inv, omega_b):
    """
    Evaluates the cost function given all parameters.
    """

    n = len(gamma)
    N = x.shape[0]

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


def objht(x, B, omega_b):
    n = len(x)
    obj = 0

    for i in range(0, n):
        obj += np.sqrt(B[i])

    obj = np.sqrt(2 * omega_b * (1 - omega_b)) * obj

    return obj


def Transient_IA(means, SCVs, omega_b, tol=None):
    """
    Computes the optimal schedule.
    wis = waiting in system. # N = n + wis
    """

    n = len(means)
    gamma, T = zip(*[phase_parameters(means[i], SCVs[i]) for i in range(n)])
    Vn = create_Vn(gamma, T)
    Vn_inv = inv(Vn)

    # assigning new variables for heavy traffic computations (equation number 2 in the writeup)
    v = [None] * n
    x = [None] * n
    B = [None] * n
    Nu = [None] * n
    De = [None] * n

    nu = 0
    de = 0
    for i in range(0, n):
        v[i] = SCVs[i] * pow(means[i], 2)

    al = 0.5

    for i in range(1, n + 1):
        for j in range(0, i):
            nu += v[j] * pow(al, i - j)
            de += pow(al, i - j)
        Nu[i - 1] = nu
        De[i - 1] = de
        B[i - 1] = nu / de  # S(i) for heavy traffic in code

    for i in range(0, n):
        x[i] = means[i] + np.sqrt(((1 - omega_b) * B[i]) / (2 * omega_b))

    # print(x)

    # minimization
    cost_fun_ht = objht(x, B, omega_b)  # heavy traffic loss function

    # true optimal loss function
    x0 = np.array([1.5] + [1.5] * (n - 1))  # initial guess, of length n
    cost_fun = lambda x: cost(x, gamma, Vn, Vn_inv, omega_b)
    lin_cons = LinearConstraint(np.eye(n), 0, np.inf)
    optim = minimize(cost_fun, x0, constraints=lin_cons, method="SLSQP", tol=tol)

    return x, cost_fun_ht, optim.x, optim.fun


n = 10
# n = n - 1

omega_b = 0.8

# means = [0.5] * 3 + [0.5] * 4 + [0.5] * 3
# SCVs = [0.5] * 3 + [0.2] * 4 + [0.5] * 3
means = [0.5] * n
SCVs = [0.5] * n

# %time Transient_IA(means, SCVs, omega)
