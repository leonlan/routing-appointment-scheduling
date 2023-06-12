import multiprocessing
from functools import partial
from itertools import permutations
from time import perf_counter

from tsp_as.appointment.heavy_traffic import compute_schedule as compute_ht_schedule
from tsp_as.appointment.true_optimal import compute_optimal_schedule
from tsp_as.classes import CostEvaluator, ProblemData, Solution

from .Result import Result


def full_enumeration(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    approx_pool_size: int = 50000,  # little more than 8!
    num_procs: int = 1,
    **kwargs,
):
    """
    Obtains the optimal solution by enumerating all possible visits.

    It is also possible to approximately enumerate: in this case, the we first
    enumerate all solutions using the heavy traffic schedule. Then, we take the
    top `approx_pool_size` solutions and evaluate them using the optimal
    schedule.

    Parameters
    ----------
    seed
        The seed for the random number generator.
    data
        The data for the problem instance.
    cost_evaluator
        The cost evaluator.
    approx_pool_size
        The size of the pool of sequences to evaluate. The pool is
        constructed by taking the top `approx_pool_size` solutions using
        the heavy traffic schedule.
    num_procs
        The number of processes to use. If 1, the search is sequential.
    """
    start = perf_counter()
    pool = [list(visits) for visits in permutations(range(1, data.dimension))]

    if approx_pool_size is not None and approx_pool_size < len(pool):
        # Filter the candidate pool of solutions using the heavy traffic
        # schedule, if the candidate pool is small enough.
        pool = _filter_using_heavy_traffic(
            pool, approx_pool_size, data, cost_evaluator, num_procs
        )

    solutions = []

    with multiprocessing.Pool(num_procs) as mp_pool:
        func = partial(_make_solution, data=data, cost_evaluator=cost_evaluator)

        for solution in mp_pool.imap_unordered(func, pool):
            solutions.append(solution)

    best = min(solutions, key=lambda s: s.cost)

    return Result(best, perf_counter() - start, 0)


def _filter_using_heavy_traffic(
    pool, approx_pool_size, data, cost_evaluator, num_procs
):
    """
    Evaluates the pool of permutations using the heavy traffic schedule, and
    return the top `approx_pool_size` solutions.
    """
    candidates = []
    with multiprocessing.Pool(num_procs) as mp_pool:
        func = partial(_make_ht_solution, data=data, cost_evaluator=cost_evaluator)

        for solution in mp_pool.imap_unordered(func, pool):
            candidates.append(solution)

    candidates.sort(key=lambda solution: solution.cost)
    return [solution.visits for solution in candidates[:approx_pool_size]]


def _make_solution(visits, data, cost_evaluator):
    schedule = compute_optimal_schedule(visits, data, cost_evaluator)
    return Solution(data, cost_evaluator, visits, schedule)


def _make_ht_solution(visits, data, cost_evaluator):
    schedule = compute_ht_schedule(visits, data, cost_evaluator)
    return Solution(data, cost_evaluator, visits, schedule)
