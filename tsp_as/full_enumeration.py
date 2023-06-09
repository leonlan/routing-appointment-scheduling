from itertools import permutations
from time import perf_counter

from tsp_as.appointment.true_optimal import compute_optimal_schedule
from tsp_as.classes import CostEvaluator, ProblemData, Solution

from .Result import Result


def full_enumeration(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    initial_solution: Solution = None,
    num_procs: int = 1,
    **kwargs
):
    """
    Obtains the optimal solution by enumerating all possible visits, and for
    each visit we compute the optimal schedule.

    Parameters
    ----------
    seed
        The seed for the random number generator.
    data
        The data for the problem instance.
    cost_evaluator
        The cost evaluator.
    initial_solution
        The initial solution to use for upper bound.
    num_procs
        The number of processes to use. If 1, the search is sequential.
    """
    start = perf_counter()

    if initial_solution is None:
        ordered_visits = list(range(1, data.dimension))
        schedule = compute_optimal_schedule(ordered_visits, data, cost_evaluator)
        initial_solution = Solution(data, cost_evaluator, ordered_visits, schedule)

    best = initial_solution

    pool = permutations(range(1, data.dimension))

    for visits in pool:
        visits = list(visits)

        # If the travel cost is already larger than the best solution, then
        # we can skip this solution to avoid expensive idle/wait computations.
        if _compute_travel_cost(visits, data, cost_evaluator) > best.cost:
            continue

        schedule = compute_optimal_schedule(visits, data, cost_evaluator)
        solution = Solution(data, cost_evaluator, visits, schedule)

        if solution.cost < best.cost:
            best = solution

    return Result(best, perf_counter() - start, 0)


def _compute_travel_cost(visits, data, cost_evaluator) -> float:
    """
    Compute the travel cost of the given visits order.
    """
    distance = data.distances[[0] + visits, visits + [0]].sum()
    return cost_evaluator.travel_weight * distance
