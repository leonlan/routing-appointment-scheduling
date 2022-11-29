import numpy as np
import numpy.random as rnd

from tsp_as.appointment import heavy_traffic_optimal, heavy_traffic_pure, true_optimal
from tsp_as.classes import Params


def main():
    rng = rnd.default_rng(1)
    params = Params.from_tsplib("instances/atsp/p43.atsp", rng=rng, max_dim=10)
    omega_b = params.omega_b
    tour = np.arange(params.dimension)

    costs = []
    for _ in range(100):
        cost_ = []
        rng.shuffle(tour)

        # Heavy traffic pure
        x, cost = heavy_traffic_pure(tour, params)
        print("HTP:", cost)
        cost_.append(cost)

        # Heavy traffic optimal
        x, cost = heavy_traffic_optimal(tour, params)
        print("HTO:", cost)
        cost_.append(cost)

        # True optimal
        x, cost = true_optimal(tour, params)
        print("TO:", cost)
        cost_.append(cost)

        costs.append(cost_)

    costs = np.array(costs)


if __name__ == "__main__":
    main()
