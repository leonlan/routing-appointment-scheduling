import elkai
from alns.Result import Result
from alns.Statistics import Statistics

from tsp_as.classes import Solution


def solve_tsp(seed, data, cost_evaluator, max_iterations=None, **kwargs):
    """
    Solves the TSP without appointment scheduling.
    """
    if max_iterations is None:
        max_iterations = 10000

    visits = elkai.solve_float_matrix(data.distances, runs=max_iterations)
    visits.remove(0)  # remove depot

    stats = Statistics()
    stats.collect_objective(0)

    schedule = [0.0] * len(visits)  # dummy schedule

    return Result(Solution(data, cost_evaluator, visits, schedule), stats)
