import numpy as np
from numpy.testing import assert_array_equal

import tsp_as.evaluations.heavy_traffic as ht
import tsp_as.evaluations.true_optimal as to


def tour2params():
    """
    Compute parameters from the passed-in tour, which are used to calculate the
    schedule.
    """
    n = 10
    omega_b = 0.8

    m_b = [0.5] * n  # mean of service time
    S_b = [0.5] * n  # scv for service time

    Z_t = np.matrix(
        "0    10     3     8     4     3     3     2     8     1 ; 10     0     9     2     7     7    10    12     2     8;  3     9     0     7     2     2     1     3     7     4; 8     2     7     0     8     5     8    10     2     6 ; 4     7     2     8     0     3      4     6     5     5;   3     7     2     5     3     0     3     5     5     2;3    10     1     8     4     3     0     2     8     4;  2    12     3     10     6     5     2     0    10     4;   8     2     7     2     5     5     8    10     0     6; 1     8     4     6     5     2     4     4     6     0"
    )

    # currently both r, rl are obtained from milp via matlab
    r = np.array(
        [0, 7, 3, 5, 9, 2, 4, 6, 1, 8]
    )  # obtain a route such that it starts and end at the depot say in this case 0th node
    rl = 24

    A = np.zeros((n, n), dtype=int)
    E_t = np.zeros((n, n), dtype=int)

    e_t = np.zeros(n)  # mean travel time based on route
    for i in range(0, n - 1):
        e_t[i] = Z_t[r[i], r[i + 1]]

    e_t[n - 1] = Z_t[r[n - 1], r[0]]

    print(e_t)

    sigm = 0.1  # a scv matrix can be added instead of sigma with randomly chosen values between
    # [0.1, 0.4] and then chosen based on the route just like e_t

    # for the other option sigma can look like sigm=[0.3 0.2 0.4 0.1 0.1 0.3 0.2 0.1 0.4 0.2]

    V_t = pow(sigm, 2) * np.power(e_t, 2)  # variance of travel matrix scv*(mean^2)

    V_b = S_b * np.power(m_b, 2)  # variance of service time

    means = np.add(e_t, m_b)
    var_sum = np.add(V_t, V_b)
    denom = np.power(means, 2)
    SCVs = np.divide(var_sum, denom)

    # from here we have means and SCVs of variable U which is sum of both travel and service times,
    # now this can be given to any of the programs via transientIA

    return means, SCVs


def main():
    omega_b = 0.8
    means, SCVs = tour2params()

    # Heavy traffic pure
    x = ht.compute_schedule(means, SCVs, omega_b)
    cost = ht.compute_objective(means, SCVs, omega_b)

    # Heavy traffic optimal
    x = ht.compute_schedule(means, SCVs, omega_b)
    # cost = to.compute_objective(x, omega_b)

    # true optimal
    # x, cost = to.compute_schedule_and_objective(...)
    #

    # Testing using Bharti's examples
    means = 0.5 * np.ones(10)
    SCVs = 0.5 * np.ones(10)
    x = ht.compute_schedule(means, SCVs, omega_b)
    cost = ht.compute_objective(means, SCVs, omega_b)

    assert_array_equal(
        x, [0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625, 0.625]
    )
    assert np.isclose(cost, 2.0)


if __name__ == "__main__":
    main()
