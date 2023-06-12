import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import poisson


class ProblemData:
    def __init__(
        self,
        name: str,
        coords: np.ndarray,
        dimension: int,
        distances: np.ndarray,
        distances_scv: np.ndarray,
        service: np.ndarray,
        service_scv: np.ndarray,
    ):
        """
        A class to represent the data of a problem instance.

        Parameters
        ----------
        name
            The name of the problem instance.
        coords
            The coordinates of the locations.
        dimension
            The number of locations.
        distances
            The distances between the locations.
        distances_scv
            The squared coefficient of variation of the distances.
        service
            The service times at the locations.
        service_scv
            The squared coefficient of variation of the service times.
        """
        self.name = name
        self.coords = coords
        self.dimension = dimension
        self.distances = distances
        self.distances_scv = distances_scv
        self.service = service
        self.service_scv = service_scv

        self.arcs_mean, self.arcs_scv, self.arcs_var = compute_arc_data(
            distances, distances_scv, service, service_scv
        )
        self.alphas, self.transitions = compute_phase_parameters(
            self.arcs_mean, self.arcs_scv
        )

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


def compute_arc_data(distances, distances_scv, service, service_scv):
    """
    Computes the means, SCVs and variances of the arc random variable, i.e.,
    the sum of the travel time and the service time.
    """
    # Compute the variances of the combined service and travel times,
    # which in turn are used to compute the scvs.
    _service_var = service_scv * (service ** 2)
    _distances_var = distances_scv * (distances ** 2)
    _var = _service_var[np.newaxis, :].T + _distances_var

    # The means and scvs are the combined service and travel times, where
    # entry (i, j) denotes the travel time from i to j and the service
    # time at location i.
    means = service[np.newaxis, :].T + distances

    with np.errstate(divide="ignore", invalid="ignore"):
        scvs = np.divide(_var, means ** 2)  # There may be NaNs in the means

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

    for i, j in product(range(n), range(n)):
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
    if SCV < 1:
        # Weighted Erlang case
        K = math.floor(1 / SCV)
        prob = ((K + 1) * SCV - math.sqrt((K + 1) * (1 - K * SCV))) / (SCV + 1)
        mu = (K + 1 - prob) / mean

        alpha = np.zeros((1, K + 1))
        B_sf = poisson.cdf(K - 1, mu) + (1 - prob) * poisson.pmf(K, mu)

        alpha[0, :] = [poisson.pmf(k, mu) / B_sf for k in range(K + 1)]
        alpha[0, K] *= 1 - prob

        transition = -mu * np.eye(K + 1)
        transition += mu * np.diag(np.ones(K), k=1)  # one above diagonal
        transition[K - 1, K] = (1 - prob) * mu
    else:
        # Hyperexponential case
        prob = (1 + np.sqrt((SCV - 1) / (SCV + 1))) / 2
        mu1 = 2 * prob / mean
        mu2 = 2 * (1 - prob) / mean

        B_sf = prob * np.exp(-mu1) + (1 - prob) * np.exp(-mu2)
        term = prob * np.exp(-mu1) / B_sf

        alpha = np.array([[term, 1 - term]])
        transition = np.diag([-mu1, -mu2])

    return alpha, transition
