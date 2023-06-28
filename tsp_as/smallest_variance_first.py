import time

from tsp_as.appointment.true_optimal import compute_optimal_schedule
from tsp_as.classes import CostEvaluator, ProblemData, Solution

from .Result import Result


def smallest_variance_first(
    seed: int, data: ProblemData, cost_evaluator: CostEvaluator, **kwargs
) -> Result:
    """
    Creates a visits in increasing order of variances.
    """
    start = time.perf_counter()

    visits: list[int] = []

    for _ in range(1, data.dimension):
        unvisited = [idx for idx in range(1, data.dimension) if idx not in visits]

        frm = 0 if len(visits) == 0 else visits[-1]
        to = unvisited[data.arcs_var[frm, unvisited].argmin()]

        visits.append(to)

    schedule = compute_optimal_schedule(data, cost_evaluator, visits)
    solution = Solution(data, cost_evaluator, visits, schedule)

    return Result(solution, time.perf_counter() - start, 0)
