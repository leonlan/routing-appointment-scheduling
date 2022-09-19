import numpy as np
import tsplib95


class Params:
    def __init__(self, name, dimension, distances):
        self.name = name
        self.dimension = dimension
        self.distances = distances

    @classmethod
    def from_tsplib(cls, path, max_dim=None):
        problem = tsplib95.load(path)

        if max_dim is None:
            max_dim = problem.dimension

        name = problem.name
        dimension = problem.dimension if max_dim is None else max_dim
        distances = np.array(problem.edge_weights)[:max_dim, :max_dim]

        # TODO Need to define all stochasticity at instantiation as well
        return cls(name, dimension, distances)
