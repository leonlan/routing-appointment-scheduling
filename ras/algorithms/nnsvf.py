import time
from itertools import product

import numpy as np

from ras.appointment.true_optimal import compute_optimal_schedule
from ras.classes import CostEvaluator, ProblemData, Result, Solution


def nnsvf(
    seed: int, data: ProblemData, cost_evaluator: CostEvaluator, **kwargs
) -> Result:
    """
    Nearest neighbor smallest variance first (NNSVF) algorithm.

    Greedily selects the next client to visit by choosing the one with the
    smallest variance of the arc, defined as the travel time plus the service
    time of the _next_ client (this is different from the $U$ variables used
    in the paper).
    """
    start = time.perf_counter()

    visits: list[int] = []
    arcs_var = compute_arcs_variance(data)

    for _ in range(1, data.dimension):
        unvisited = [idx for idx in range(1, data.dimension) if idx not in visits]

        frm = 0 if len(visits) == 0 else visits[-1]
        to = unvisited[arcs_var[frm, unvisited].argmin()]

        visits.append(to)

    schedule = compute_optimal_schedule(data, cost_evaluator, visits)
    solution = Solution(data, cost_evaluator, visits, schedule)

    return Result(solution, time.perf_counter() - start, 0)


def compute_arcs_variance(data):
    _distances_var = data.distances_scv * (data.distances**2)
    arc_variance = np.zeros(_distances_var.shape)

    for i, j in product(range(data.dimension), repeat=2):
        arc_variance[i, j] = _distances_var[i, j] + data.service_var[j]

    return arc_variance
