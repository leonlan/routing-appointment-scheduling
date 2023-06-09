import time

from tsp_as.appointment.true_optimal import compute_optimal_schedule
from tsp_as.classes import Solution

from .Result import Result


def smallest_variance_first(seed, data, cost_evaluator, **kwargs):
    """
    Creates a visits in increasing order of variances.
    """
    start = time.perf_counter()

    service_var = data.service_scv * data.service**2
    visits = service_var.argsort().tolist()
    visits.remove(0)  # ignore the depot

    schedule = compute_optimal_schedule(visits, data, cost_evaluator)
    solution = Solution(data, cost_evaluator, visits, schedule)

    return Result(solution, time.perf_counter() - start, 0)
