import numpy as np
import numpy.random as rnd
from numpy.testing import assert_array_almost_equal, assert_array_equal

import tsp_as.evaluations.heavy_traffic as ht
import tsp_as.evaluations.true_optimal as to


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
    scv_travel = (0.1 * np.ones(n)) ** 2  # TODO remove this when not testing
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
    tour = np.array([0, 7, 3, 5, 9, 2, 4, 6, 1, 8])
    dist = np.matrix(
        "0    10     3     8     4     3     3     2     8     1 ; 10     0     9     2     7     7    10    12     2     8;  3     9     0     7     2     2     1     3     7     4; 8     2     7     0     8     5     8    10     2     6 ; 4     7     2     8     0     3      4     6     5     5;   3     7     2     5     3     0     3     5     5     2;3    10     1     8     4     3     0     2     8     4;  2    12     3     10     6     5     2     0    10     4;   8     2     7     2     5     5     8    10     0     6; 1     8     4     6     5     2     4     4     6     0"
    )
    means, SCVs = tour2params(tour, dist, rng)

    test()

    # Heavy traffic pure
    x = ht.compute_schedule(means, SCVs, omega_b)
    cost = ht.compute_objective(means, SCVs, omega_b)
    print(cost)

    # Heavy traffic optimal
    x = ht.compute_schedule(means, SCVs, omega_b)
    cost = to.compute_objective_(x, means, SCVs, omega_b)
    print(cost)

    # True optimal
    # x, cost = to.compute_schedule(means, SCVs, omega_b, tol=1e-2)
    print(cost)


if __name__ == "__main__":
    main()
