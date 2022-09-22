import numpy as np


def tour2params(tour, params):
    """
    Compute parameters from the passed-in tour, which are used to calculate the
    schedule.
    """
    n = len(tour)

    mean_service = params.service[tour]
    scv_service = params.service_scv[tour]
    var_service = scv_service * np.power(mean_service, 2)

    mean_travel = np.array(
        [params.distances[tour[i], tour[(i + 1) % n]] for i in range(n)]
    )
    scv_travel = np.array(
        [params.distances_scv[tour[i], tour[(i + 1) % n]] for i in range(n)]
    )
    var_travel = scv_travel * np.power(mean_travel, 2)

    mean_sum = mean_travel + mean_service
    var_sum = var_travel + var_service
    denom = np.power(mean_sum, 2)
    SCVs = np.divide(var_sum, denom)

    return mean_sum, SCVs
