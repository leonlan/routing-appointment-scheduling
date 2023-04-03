import math

import numpy as np
import numpy.random as rnd
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
        omega_travel=0.2,
        omega_idle=0.2,
        omega_wait=0.6,
        lag=3,
    ):
        self.name = name
        self.coords = coords
        self.dimension = dimension
        self.distances = distances
        self.distances_scv = distances_scv
        self.service = service
        self.service_scv = service_scv
        self.means, self.scvs = compute_means_scvs(
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
        path = Path(loc)
        # TODO

        return cls(
            path.stem,
            distances,
            distances_scv,
            service,
            service_scv,
            **kwargs,
        )

    @classmethod
    def make_random(
        cls,
        seed,
        dim,
        max_size,
        max_service_time,
        distances_scv_min=0.1,
        distances_scv_max=0.1,
        service_scv_min=1.1,
        service_scv_max=1.5,
        name=None,
        **kwargs,
    ):
        """
        Creates a random instance with ``dimension`` locations.

        Customer locations are randomly sampled from a grid of size `max_size`.
        The Euclidean distances are computed for the distances, and service
        times are drawn uniformly between one and `max_service_time`.
        """
        rng = rnd.default_rng(seed)
        name = "Random instance." if name is None else name
        coords = rng.integers(max_size, size=(dim, dim))

        distances = pairwise_euclidean(coords)
        distances_scv = rng.uniform(
            low=distances_scv_min,
            high=distances_scv_max,
            size=distances.shape,
        )

        service = rng.integers(max_service_time, size=dim) + 1  # at least one
        service_scv = rng.uniform(
            low=service_scv_min,
            high=service_scv_max,
            size=service.shape,
        )

        return cls(
            name,
            coords,
            dim,
            distances,
            distances_scv,
            service,
            service_scv,
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

    return means, scvs


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


def pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise Euclidean distance between the passed-in coordinates.

    Parameters
    ----------
    coords
        An n-by-2 array of location coordinates.

    Returns
    -------
    np.ndarray
        An n-by-n Euclidean distances matrix.

    """
    # Subtract each coordinate from every other coordinate
    diff = coords[:, np.newaxis, :] - coords
    square_diff = diff**2
    square_dist = np.sum(square_diff, axis=-1)
    return np.sqrt(square_dist)
