import numpy as np
import numpy.random as rnd
from numpy.testing import assert_array_almost_equal, assert_array_equal

import tsp_as.evaluations.heavy_traffic as ht
import tsp_as.evaluations.true_optimal as to
from tsp_as.classes import Params, Solution


def tour2params(tour, dist, rng):
    """
    Compute parameters from the passed-in tour, which are used to calculate the
    schedule.
    """
    n = len(tour)

    mean_service = 0.5 * np.ones(n)
    scv_service = 0.5 * np.ones(n)
    var_service = scv_service * np.power(mean_service, 2)

    mean_travel = np.array([dist[tour[i], tour[(i + 1) % n]] for i in range(n)])
    scv_travel = rng.uniform(low=0.1, high=0.4, size=n)
    var_travel = scv_travel * np.power(mean_travel, 2)

    mean_sum = mean_travel + mean_service
    var_sum = var_travel + var_service
    denom = np.power(mean_sum, 2)
    SCVs = np.divide(var_sum, denom)

    return mean_sum, SCVs


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

    params = Params.from_tsplib("instances/atsp/br17.atsp", max_dim=10)
    tour = np.arange(params.dimension)

    for _ in range(10):
        rng.shuffle(tour)
        means, SCVs = tour2params(tour, params.distances, rng)

        test()

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
