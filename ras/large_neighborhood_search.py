import time
from typing import Optional

import numpy.random as rnd
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations, MaxRuntime

from ras.appointment.heavy_traffic import compute_schedule as compute_ht_schedule
from ras.appointment.true_optimal import compute_optimal_schedule
from ras.classes import CostEvaluator, ProblemData, Solution
from ras.operators import adjacent_destroy, greedy_insert, random_destroy

from .Result import Result


def large_neighborhood_search(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    initial_visits: Optional[list[int]] = None,
    max_runtime: Optional[float] = None,
    max_iterations: Optional[int] = None,
    **kwargs,
) -> Result:
    """
    Solve the appointment scheduling problem using the LNS metaheuristic.

    Parameters
    ----------
    seed
        The random seed to use.
    data
        The problem data.
    cost_evaluator
        The cost evaluator.
    initial_visits
        The visits of the initial solution. If None, a random visits are used.
    max_runtime
        The maximum runtime in seconds. If None, max_iterations must be specified.
    max_iterations
        The maximum number of iterations. If None, max_runtime must be specified.
    **kwargs
        Additional keyword arguments to pass to the ALNS solver.
    """
    start = time.perf_counter()
    rng = rnd.default_rng(seed)

    alns = ALNS(rng)

    D_OPS = [
        adjacent_destroy,
        random_destroy,
    ]
    for d_op in D_OPS:
        alns.add_destroy_operator(d_op)

    R_OPS = [greedy_insert]
    for r_op in R_OPS:
        alns.add_repair_operator(r_op)

    # We explicitly take initial visits because we want to be able to
    # compute the heavy traffic schedule before the ALNS starts. This gives
    # a "fair" objective value to compare with.
    if initial_visits is None:
        initial_visits = rng.permutation(list(range(1, data.dimension))).tolist()

    ht_schedule = compute_ht_schedule(data, cost_evaluator, initial_visits)
    init = Solution(data, cost_evaluator, initial_visits, ht_schedule)

    select = RouletteWheel([5, 2, 1, 0.5], 0.5, len(D_OPS), len(R_OPS))

    # In percentages of the intial solution.
    start_threshold_pct = kwargs.get("rrt_start_threshold_pct", 0.05)
    end_threshold_pct = 0

    if max_runtime is not None:
        stop = MaxRuntime(max_runtime)
        start_threshold = init.objective() * start_threshold_pct
        end_threshold = init.objective() * end_threshold_pct
        accept = TimeBasedRecordToRecordTravel(
            start_threshold, end_threshold, max_runtime
        )
    else:
        assert max_iterations is not None
        stop = MaxIterations(max_iterations)
        accept = RecordToRecordTravel.autofit(
            init.objective(), start_threshold_pct, end_threshold_pct, max_iterations
        )

    alns_result = alns.iterate(
        init, select, accept, stop, data=data, cost_evaluator=cost_evaluator, **kwargs
    )

    visits = alns_result.best_state.visits
    schedule = compute_optimal_schedule(data, cost_evaluator, visits)
    solution = Solution(data, cost_evaluator, visits, schedule)

    return Result(
        solution, time.perf_counter() - start, len(alns_result.statistics.runtimes)
    )


def time_based_value(
    start_value: float,
    end_value: float,
    max_runtime: float,
    start_time: float,
    method: str,
):
    if max_runtime < 0:
        raise ValueError("max_runtime must be >= 0")

    if start_value < end_value:
        raise ValueError("start_value must be >= end_value")

    if end_value < 0 and method == "exponential":
        raise ValueError("end_value must be > 0 for exponential method")

    if max_runtime == 0:
        return end_value

    # pct_elapsed_time=0 should give the start temperature,
    # pct_elapsed_time=1 should give the end temperature.
    pct_elapsed_time = (time.perf_counter() - start_time) / max_runtime
    pct_elapsed_time = min(pct_elapsed_time, 1)

    if method == "linear":
        delta = start_value - end_value
        return start_value - delta * pct_elapsed_time
    elif method == "exponential":
        fraction = end_value / start_value
        return start_value * fraction**pct_elapsed_time
    else:
        raise ValueError(f"Method {method} not known.")


class TimeBasedRecordToRecordTravel:
    def __init__(
        self,
        start_threshold: float,
        end_threshold: float,
        max_runtime: float,
        method: str = "linear",
        cmp_best: bool = True,
    ):
        self._start_threshold = start_threshold
        self._end_threshold = end_threshold
        self._max_runtime = max_runtime
        self._method = method
        self._cmp_best = cmp_best

        self._delta_threshold = start_threshold - end_threshold
        self._start_time = time.perf_counter()

    @property
    def start_threshold(self) -> float:
        return self._start_threshold

    @property
    def end_threshold(self) -> float:
        return self._end_threshold

    @property
    def max_runtime(self) -> float:
        return self._max_runtime

    @property
    def method(self) -> str:
        return self._method

    def __call__(self, rnd, best, curr, cand):
        threshold = time_based_value(
            self.start_threshold,
            self.end_threshold,
            self.max_runtime,
            self._start_time,
            self.method,
        )
        baseline = best if self._cmp_best else curr
        return cand.objective() - baseline.objective() <= threshold
