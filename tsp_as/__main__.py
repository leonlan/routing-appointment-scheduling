import numpy as np
import numpy.random as rnd
from numpy.testing import assert_array_almost_equal, assert_array_equal

import tsp_as.evaluations.heavy_traffic as ht
import tsp_as.evaluations.true_optimal as to
from tsp_as.classes import Params, Solution
from tsp_as.evaluations.tour2params import tour2params


def test():
    # Testing using Bharti's examples
    omega_b = 0.8
    means = 0.5 * np.ones(10)
    SCVs = 0.5 * np.ones(10)
    x = ht.compute_schedule(means, SCVs, omega_b)
    cost = ht.compute_objective(means, SCVs, omega_b)

    # Heavy traffic pure
    assert_array_equal(
        x, [0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625]
    )
    assert np.isclose(cost, 2.0)

    # True optimal
    x, cost = to.compute_schedule(means, SCVs, omega_b)

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
    omega_b = 0.8

    rng = rnd.default_rng(1)

    params = Params.from_tsplib("instances/atsp/br17.atsp", rng=rng, max_dim=10)
    tour = np.arange(params.dimension)

    test()
    for _ in range(100):
        rng.shuffle(tour)
        means, SCVs = tour2params(tour, params, rng)

        # Heavy traffic pure
        x = ht.compute_schedule(means, SCVs, omega_b)
        cost = ht.compute_objective(means, SCVs, omega_b)
        print("HTP:", cost)

        # Heavy traffic optimal
        x = ht.compute_schedule(means, SCVs, omega_b)
        cost = to.compute_objective_(x, means, SCVs, omega_b)
        print("HTO:", cost)

        # True optimal
        x, cost = to.compute_schedule(means, SCVs, omega_b, tol=1e-2)
        print("TO:", cost)


if __name__ == "__main__":
    main()
