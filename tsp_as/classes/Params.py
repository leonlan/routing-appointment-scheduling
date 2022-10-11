import numpy as np
import tsplib95


class Params:
    def __init__(self, name, rng, dimension, distances, G, **kwargs):
        self.name = name
        self.dimension = dimension
        self.distances = distances
        self.distances_scv = rng.uniform(
            low=kwargs.get("distances_csv_min", 0.1),
            high=kwargs.get("distances_csv_max", 0.5),
            size=distances.shape,
        )
        self.G = G

        self.service = 0.5 * np.ones(dimension)  # TODO How to determine this?
        self.service_scv = 0.5 * np.ones(dimension)  # TODO How to determine this?

        self.omega = kwargs.get("omega", 0.0)
        self.omega_b = kwargs.get("omega_", 0.8)

        self.objective = kwargs.get("objective", "hto")

        # TODO Remove this later
        self.trajectory = []

    @classmethod
    def from_tsplib(cls, path, rng, **kwargs):
        problem = tsplib95.load(path)

        name = problem.name
        dimension = min(problem.dimension, kwargs.get("max_dim", problem.dimension))

        distances = np.array(problem.edge_weights)[:dimension, :dimension]
        G = problem.get_graph()  # Used for plotting

        return cls(name, rng, dimension, distances, G, **kwargs)
