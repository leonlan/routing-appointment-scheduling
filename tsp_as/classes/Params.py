import numpy as np
import tsplib95


class Params:
    def __init__(self, name, rng, dimension, distances):
        self.name = name
        self.dimension = dimension
        self.omega = 0.0
        self.omega_b = 0.8
        self.distances = distances
        self.distances_scv = rng.uniform(low=0.1, high=0.4, size=distances.shape)
        self.service = 0.5 * np.ones(dimension)
        self.service_scv = 0.5 * np.ones(dimension)

        self.objective = "htp"

    @classmethod
    def from_tsplib(cls, path, rng, max_dim=None):
        problem = tsplib95.load(path)

        name = problem.name
        dimension = (
            problem.dimension if max_dim is None else min(problem.dimension, max_dim)
        )
        distances = np.array(problem.edge_weights)[:dimension, :dimension]

        # TODO Need to define all stochasticity at instantiation as well
        return cls(name, rng, dimension, distances)
