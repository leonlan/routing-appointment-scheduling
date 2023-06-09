import multiprocessing
from functools import partial
from itertools import permutations
from time import perf_counter
from typing import Optional

from tsp_as.appointment.heavy_traffic import compute_schedule as compute_ht_schedule
from tsp_as.appointment.true_optimal import compute_optimal_schedule
from tsp_as.classes import CostEvaluator, ProblemData, Solution

from .Result import Result


def full_enumeration(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    initial_solution: Optional[Solution] = None,
    approx_pool_size: Optional[int] = 100,
    num_procs: int = 8,
    **kwargs,
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
        The initial to solution use for upper bound.
    approx_pool_size
        The size of the pool of permutations to evaluate. The pool is
        constructed by taking the top `approx_pool_size` solutions using
        the heavy traffic schedule.
    num_procs
        The number of processes to use. If 1, the search is sequential.
    """
    start = perf_counter()

    if initial_solution is None:
        ordered_visits = list(range(1, data.dimension))
        schedule = compute_optimal_schedule(ordered_visits, data, cost_evaluator)
        initial_solution = Solution(data, cost_evaluator, ordered_visits, schedule)

    pool = [list(visits) for visits in permutations(range(1, data.dimension))]

    if approx_pool_size is not None:
        pool = _filter_using_heavy_traffic(pool, approx_pool_size, data, cost_evaluator)

    solutions = []
    with multiprocessing.Pool(num_procs) as mp_pool:
        func = partial(
            _make_solution,
            data=data,
            cost_evaluator=cost_evaluator,
            upper_bound=initial_solution.cost,
        )

        for solution in mp_pool.imap_unordered(func, pool):
            if solution is not None:
                solutions.append(solution)

    best = min(solutions, key=lambda s: s.cost)

    return Result(best, perf_counter() - start, 0)


def _filter_using_heavy_traffic(pool, approx_pool_size, data, cost_evaluator):
    """
    Evaluates the pool of permutations using the heavy traffic schedule, and
    return the top `approx_pool_size` solutions.
    """

    def compute_approx_cost(visits):
        schedule = compute_ht_schedule(visits, data, cost_evaluator)
        return cost_evaluator(visits, schedule, data)

    pool.sort(key=lambda visits: compute_approx_cost(visits))
    return pool[:approx_pool_size]


def _make_solution(visits, data, cost_evaluator, upper_bound):
    if _compute_travel_cost(visits, data, cost_evaluator) > upper_bound:
        return None

    schedule = compute_optimal_schedule(visits, data, cost_evaluator)
    solution = Solution(data, cost_evaluator, visits, schedule)

    return solution


def _compute_travel_cost(visits, data, cost_evaluator) -> float:
    distance = data.distances[[0] + visits, visits + [0]].sum()
    return cost_evaluator.travel_weight * distance
