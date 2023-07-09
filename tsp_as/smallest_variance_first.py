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

    visits = data.service_var.argsort().tolist()
    visits.remove(0)

    schedule = compute_optimal_schedule(data, cost_evaluator, visits)
    solution = Solution(data, cost_evaluator, visits, schedule)

    return Result(solution, time.perf_counter() - start, 0)
