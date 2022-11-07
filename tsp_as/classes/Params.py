from pathlib import Path

import numpy as np


class Params:
    def __init__(self, name, rng, dimension, distances, **kwargs):
        self.name = name
        self.dimension = dimension
        self.distances = distances
        self.distances_scv = rng.uniform(
            low=kwargs.get("distances_csv_min", 0.1),
            high=kwargs.get("distances_csv_max", 0.5),
            size=distances.shape,
        )

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
        Read an ATSP instance from TSPLIB.
        """
        path = Path(loc)

        with open(path, "r") as fi:
            data = fi.readlines()

        for line in data:
            if line.startswith("DIMENSION:"):
                n = int(line.split()[1])
                break

        # k is the line that edge weight section starts
        for k, line in enumerate(data):
            if line.startswith("EDGE_WEIGHT_SECTION"):
                break

        # flatten list of distances
        dist = []
        for line in data[k + 1 :]:
            if line.startswith("EOF"):
                break

            for val in line.split():
                dist.append(int(val))

        distances = np.reshape(dist, (n, n))

        name = path.stem
        # Possibly reduce the dimensions of the data
        dimension = min(n, kwargs.get("max_dim", n))
        distances = distances[:dimension, :dimension]

        return cls(name, rng, dimension, distances, **kwargs)
