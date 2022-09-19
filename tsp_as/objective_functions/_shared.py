import math

import numpy as np
from scipy.linalg import inv  # matrix inversion
from scipy.linalg.blas import dgemm, dgemv  # matrix multiplication
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse.linalg import expm  # matrix exponential
from scipy.stats import poisson


def phase_parameters(mean, SCV):
    """
    Returns the initial distribution gamma and the transition rate
    matrix T of the phase-fitted service times given the mean, SCV,
    and the elapsed service time u of the client in service.
    """

    # weighted Erlang case
    if SCV < 1:

        # parameters
        K = math.floor(1 / SCV)
        p = ((K + 1) * SCV - math.sqrt((K + 1) * (1 - K * SCV))) / (SCV + 1)
        mu = (K + 1 - p) / mean

        # initial distribution
        gamma_i = np.zeros((1, K + 1))
        B_sf = poisson.cdf(K - 1, mu) + (1 - p) * poisson.pmf(K, mu)
        for z in range(K + 1):
            gamma_i[0, z] = poisson.pmf(z, mu) / B_sf
        gamma_i[0, K] *= 1 - p

        # transition rate matrix
        Ti = -mu * np.eye(K + 1)
        for i in range(K - 1):
            Ti[i, i + 1] = mu
        Ti[K - 1, K] = (1 - p) * mu

    # hyperexponential case
    else:

        # parameters
        p = (1 + np.sqrt((SCV - 1) / (SCV + 1))) / 2
        mu1 = 2 * p / mean
        mu2 = 2 * (1 - p) / mean

        # initial distribution
        gamma_i = np.zeros((1, 2))
        B_sf = p * np.exp(-mu1) + (1 - p) * np.exp(-mu2)
        gamma_i[0, 0] = p * np.exp(-mu1) / B_sf
        gamma_i[0, 1] = 1 - gamma_i[0, 0]

        # transition rate matrix
        Ti = np.zeros((2, 2))
        Ti[0, 0] = -mu1
        Ti[1, 1] = -mu2

    return gamma_i, Ti


def create_Vn(gamma, T):
    """
    Creates the matrix Vn given the
    initial distributions gamma and the
    corresponding transition matrices T.
    """

    # initialize Vn
    n = len(T)
    d = [T[i].shape[0] for i in range(n)]
    dim_V = np.cumsum([0] + d)
    Vn = np.zeros((dim_V[n], dim_V[n]))

    # compute Vn recursively
    for i in range(1, n):
        Vn[dim_V[i - 1] : dim_V[i], dim_V[i - 1] : dim_V[i]] = T[i - 1]
        Vn[dim_V[i - 1] : dim_V[i], dim_V[i] : dim_V[i + 1]] = (
            np.matrix(-T[i - 1] @ np.ones((d[i - 1], 1))) @ gamma[i]
        )

    Vn[dim_V[n - 1] : dim_V[n], dim_V[n - 1] : dim_V[n]] = T[n - 1]

    return Vn
