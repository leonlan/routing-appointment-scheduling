import time

import numpy as np
import numpy.random as rnd
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations, MaxRuntime

from tsp_as.classes import Solution
from tsp_as.destroy_operators import adjacent_destroy, random_destroy
from tsp_as.repair_operators import greedy_insert


def solve_alns(
    seed,
    data,
    cost_evaluator,
    init=None,
    max_runtime=None,
    max_iterations=None,
    **kwargs,
):
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

    if init is None:
        ordered = np.arange(1, data.dimension).tolist()
        init = Solution(data, cost_evaluator, ordered)

    select = RouletteWheel([5, 2, 1, 0.5], 0.5, len(D_OPS), len(R_OPS))

    start_threshold = 0.05
    end_threshold = 0

    if max_runtime is not None:
        stop = MaxRuntime(max_runtime)
        accept = TimeBasedRecordToRecordTravel(
            start_threshold, end_threshold, max_runtime
        )
    else:
        assert max_iterations is not None
        stop = MaxIterations(max_iterations)
        accept = RecordToRecordTravel.autofit(
            init.objective(), start_threshold, end_threshold, max_iterations
        )

    return alns.iterate(init, select, accept, stop, **kwargs)


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
