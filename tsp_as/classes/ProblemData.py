import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import poisson


class ProblemData:
    def __init__(
        self,
        name,
        coords,
        dimension,
        distances,
        distances_scv,
        service,
        service_scv,
        objective="hto",
        omega_travel=0.5,
        omega_idle=0.25,
        omega_wait=0.25,
        lag=4,
        **kwargs,
    ):
        # distances = distances / 10
        self.name = name
        self.coords = coords
        self.dimension = dimension
        self.distances = distances
        self.distances_scv = distances_scv
        self.service = service
        self.service_scv = service_scv
        self.service_var = service_scv * np.power(service, 2)
        self.means, self.scvs, self.vars = compute_means_scvs(
            distances, distances_scv, service, service_scv
        )
        self.alphas, self.transitions = compute_phase_parameters(self.means, self.scvs)

        self.objective = objective
        self.omega_travel = omega_travel
        self.omega_idle = omega_idle
        self.omega_wait = omega_wait
        self.lag = lag  # TODO this is an algorithm parameter

    @classmethod
    def from_file(cls, loc, **kwargs):
        """
        Loads a ProblemData instances from file. The instance is assumed to
        be a JSON file, containing an array for the distances, distances SCVs,
        service times, and service times SCVs.
        """
        path = Path(loc)

        with open(path, "r") as fh:
            data = json.load(fh)
            data = {key: np.array(arr) for key, arr in data.items()}

        return cls(
            name=path.stem,
            **data,
            **kwargs,
        )


def compute_means_scvs(distances, distances_scv, service, service_scv):
    """
    Computes the means and SCVs of the random variable that is the sum of
    the travel time and the service time.
    """
    # Compute the variances of the combined service and travel times,
    # which in turn are used to compute the scvs.
    _service_var = service_scv * np.power(service, 2)
    _distances_var = distances_scv * np.power(distances, 2)
    _var = _service_var[np.newaxis, :].T + _distances_var

    # The means and scvs are the combined service and travel times, where
    # entry (i, j) denotes the travel time from i to j and the service
    # time at location i.
    means = service[np.newaxis, :].T + distances
    scvs = np.divide(_var, np.power(means, 2))

    np.fill_diagonal(means, 0)
    np.fill_diagonal(scvs, 0)

    return means, scvs, _var


def compute_phase_parameters(means, scvs):
    """
    Wrapper for `_compute_phase_parameters` to return a full matrix
    of alphas and transition matrices for the given means and scvs.
    """
    n = means.shape[0]
    alphas = np.zeros((n, n), dtype=object)
    transitions = np.zeros((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            alpha, transition = _compute_phase_parameters(means[i, j], scvs[i, j])
            alphas[i, j] = alpha
            transitions[i, j] = transition

    return alphas, transitions


def _compute_phase_parameters(mean, SCV):
    """
    Returns the initial distribution alpha and the transition rate
    matrix T of the phase-fitted service times given the mean, SCV,
    and the elapsed service time u of the client in service.
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

        alpha = np.array([[term, 1 - term]])
        transition = np.diag([-mu1, -mu2])

    return alpha, transition
