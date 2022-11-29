from itertools import combinations
from pathlib import Path

import numpy as np
import tsplib95


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
        self.coords = coords

        self.service = 0.5 * np.ones(dimension)  # TODO How to determine this?
        self.service_scv = 0.5 * np.ones(dimension)  # TODO How to determine this?

        self.omega = kwargs.get("omega", 0.0)
        self.omega_b = kwargs.get("omega_", 0.8)

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
