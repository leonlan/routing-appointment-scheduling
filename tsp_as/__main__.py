import numpy as np
import numpy.random as rnd
from numpy.testing import assert_array_almost_equal, assert_array_equal

from tsp_as.classes import Params
from tsp_as.evaluations import (heavy_traffic_optimal, heavy_traffic_pure,
                                true_optimal)
from tsp_as.evaluations.heavy_traffic import \
    compute_objective as ht_compute_objective
from tsp_as.evaluations.heavy_traffic import \
    compute_schedule as ht_compute_schedule
from tsp_as.evaluations.true_optimal import \
    compute_schedule as to_compute_schedule


def test():
    # Testing using Bharti's examples
    omega_b = 0.8
    means = 0.5 * np.ones(10)
    SCVs = 0.5 * np.ones(10)
    x = ht_compute_schedule(means, SCVs, omega_b)
    cost = ht_compute_objective(means, SCVs, omega_b)

    # Heavy traffic pure
    assert_array_equal(
        x, [0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625]
    )
    assert np.isclose(cost, 2.0)

    # True optimal
    x, cost = to_compute_schedule(means, SCVs, omega_b)

    assert_array_almost_equal(
        x,
        [
            0.12324929,
            0.29232458,
            0.33362237,
            0.34655884,
            0.35010092,
            0.34723355,
            0.33850544,
            0.32247991,
            0.29126939,
            0.21868121,
        ],
    )
    assert np.isclose(cost, 1.1475057306039138)


def main():

    rng = rnd.default_rng(1)
    params = Params.from_tsplib("instances/atsp/p43.atsp", rng=rng, max_dim=10)
    omega_b = params.omega_b
    tour = np.arange(params.dimension)

    test()
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
