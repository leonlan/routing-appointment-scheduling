import json
from itertools import product
from pathlib import Path

import numpy as np

from ras.distributions import fit_hyperexponential, fit_mixed_erlang


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
        self.service_var = service_scv * (service**2)

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

            # Scale distances for the numerical experiment that investigates
            # how the proportion of service/travel times affects performance
            data["distances"] *= kwargs.get("distance_scaling", 1)

        return cls(
            name=path.stem,
            **data,
        )


def compute_arc_data(distances, distances_scv, service, service_scv):
    """
    Computes the means, SCVs and variances of the arc random variable, i.e.,
    the sum of the travel time and the service time.
    """
    # Compute the variances of the combined service and travel times,
    # which in turn are used to compute the scvs.
    _service_var = service_scv * (service**2)
    _distances_var = distances_scv * (distances**2)
    variances = _service_var[np.newaxis, :].T + _distances_var

    # The means and scvs are the combined service and travel times, where
    # entry (i, j) denotes the travel time from i to j and the service
    # time at location i.
    means = service[np.newaxis, :].T + distances

    with np.errstate(divide="ignore", invalid="ignore"):
        scvs = np.divide(variances, means**2)  # There may be NaNs in the means
        # Round to 3 decimals to avoid floating point precision issues in fitting mixed erlang
        scvs = np.round(scvs, 3)

    np.fill_diagonal(means, 0)
    np.fill_diagonal(scvs, 0)

    return means, scvs, variances


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


def _compute_phase_parameters(mean: float, scv: float):
    """
    Returns the initial distribution alpha and the transition rate
    matrix T of the phase-fitted service times given the mean and scv.

    Based on the scv, we either fit a mixed Erlang or a hyperexponential
    distribution such that they match the first and second moment of the
    given distribution.
    """
    if scv < 1:  # Mixed Erlang case
        K, prob, mu = fit_mixed_erlang(mean, scv)

        alpha = np.zeros((1, K))
        alpha[0, 0] = 1 - prob
        alpha[0, 1] = prob

        transition = -mu * np.eye(K)
        transition += mu * np.diag(np.ones(K - 1), k=1)  # one above diagonal
        transition[K - 2, K - 1] = (1 - prob) * mu  # last row
    else:  # Hyperexponential case
        prob, mu1, mu2 = fit_hyperexponential(mean, scv)
        alpha = np.array([[prob, 1 - prob]])
        transition = np.diag([-mu1, -mu2])

    return alpha, transition
