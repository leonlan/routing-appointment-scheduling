import time

import gurobipy as gp
from numpy.random import default_rng

from ras.classes import CostEvaluator, ProblemData
from ras.Result import Result


def tsang(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    max_runtime: int,
    num_scenarios: int = 50,  # TODO remove
    **kwargs,
) -> Result:
    """
    Solves the routing and appointment scheduling problem using the sample
    average approximation method from Tsang and Shehadeh (2022).

    Parameters
    ----------
    seed
        The random seed to use for the solver (unused).
    data
        The problem data.
    cost_evaluator
        The cost evaluator to use for computing the cost of a schedule.
    max_runtime
        The maximum runtime.
    num_scenarios
        The number of scenarios to use for the sample average approximation.

    Returns
    -------
    Result
        The algorithm results.
    """
    start = time.perf_counter()

    gp.Model()

    default_rng(seed)

    return Result(solution, time.perf_counter() - start, 0)
