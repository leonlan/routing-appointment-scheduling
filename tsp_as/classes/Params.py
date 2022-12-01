import math
from itertools import combinations
from pathlib import Path

import numpy as np
import tsplib95
from scipy.stats import poisson


class Params:
    def __init__(self, name, rng, dimension, distances, coords, **kwargs):
        self.name = name
        self.dimension = dimension
        self.coords = coords

        self.distances = distances
        self.distances_scv = rng.uniform(
            low=kwargs.get("distances_scv_min", 0.1),
            high=kwargs.get("distances_scv_max", 1.5),
            size=distances.shape,
        )
        self.distances_var = self.distances_scv * np.power(self.distances, 2)

        # Mean service time is given as the average travel time to the
        # 10 closest customers
        self.service = np.append(
            [0], np.sort(self.distances, axis=1)[1:, :10].mean(axis=1)
        )
        self.service_scv = rng.uniform(
            low=kwargs.get("service_scv_min", 0.1),
            high=kwargs.get("service_scv_max", 1.5),
            size=self.service.shape,
        )
        self.service_var = self.service_scv * np.power(self.service, 2)

        # Combined travel and service times
        self.means = self.service[np.newaxis, :].T + self.distances
        self.var = self.service_var[np.newaxis, :].T + self.distances_var
        self.scvs = np.divide(self.var, np.power(self.means, 2))

        n = self.scvs.shape[0]
        self.alphas = np.zeros((n, n), dtype=object)
        self.transitions = np.zeros((n, n), dtype=object)

        for i in range(n):
            for j in range(n):
                if i != j:
                    alpha, transition = compute_phase_parameters(
                        self.means[i, j], self.scvs[i, j]
                    )
                    self.alphas[i, j] = alpha
                    self.transitions[i, j] = transition

        self.omega_travel = kwargs.get("omega_travel", 0.1)
        self.omega_idle = kwargs.get("omega_idle", 0.1)
        self.omega_wait = kwargs.get("omega_wait", 0.8)

        self.objective = kwargs.get("objective", "hto")
        self.lag = 4

        # TODO Remove this later
        self.trajectory = []

    @classmethod
    def from_tsplib(cls, loc, rng, **kwargs):
        """
        Reads a TSP instance from TSPLIB.
        """
        path = Path(loc)
        problem = tsplib95.load(path)

        name = path.stem
        name = problem.name
        dimension = min(problem.dimension, kwargs.get("max_dim", problem.dimension))

        # We need to explicitly retrieve the dimensions ourselves
        distances = np.zeros((dimension, dimension))
        for i, j in combinations(range(dimension), r=2):

            # The distance indices depends on the edge weight formats.
            shift = 0 if problem.edge_weight_type == "EXPLICIT" else 1
            d_ij = problem.get_weight(i + shift, j + shift)

            distances[i, j] = d_ij
            distances[j, i] = d_ij

        coords = np.array([coord for coord in problem.node_coords.values()])[:dimension]

        return cls(name, rng, dimension, distances, coords, **kwargs)


def compute_phase_parameters(mean, SCV):
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
