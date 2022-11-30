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
        self.distances = distances
        self.distances_scv = rng.uniform(
            low=kwargs.get("distances_csv_min", 0.1),
            high=kwargs.get("distances_csv_max", 0.5),
            size=distances.shape,
        )
        self.distances_var = self.distances_scv * np.power(self.distances, 2)

        self.coords = coords

        self.service = np.append([0], 0.5 * np.ones(dimension - 1))
        self.service_scv = np.append([0], 0.5 * np.ones(dimension - 1))
        self.service_var = self.service_scv * np.power(self.service, 2)

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

        self.omega_travel = kwargs.get("omega_travel", 0.0)
        self.omega_idle = kwargs.get("omega_idle", 0.8)
        self.omega_wait = kwargs.get("omega_wait", 0.8)
        self.omega_b = kwargs.get("omega_b", 0.8)

        self.objective = kwargs.get("objective", "hto")

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
            d_ij = problem.get_weight(i + 1, j + 1)  # start at idx 1
            distances[i, j] = d_ij
            distances[j, i] = d_ij

        coords = np.array([coord for coord in problem.node_coords.values()])[:dimension]

        return cls(name, rng, dimension, distances, coords, **kwargs)


def compute_phase_parameters(mean, SCV):
    """
    Returns the initial distribution alpha and the transition rate
    matrix T of the phase-fitted service times given the mean, SCV,
    and the elapsed service time u of the client in service.

    # TODO These phase parameters can be computed in params as well?
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

        alpha = np.array([term, 1 - term])
        transition = np.diag([-mu1, -mu2])

    return alpha, transition
