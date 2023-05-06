import numpy as np
import numpy.random as rnd
from alns import ALNS
from alns.accept import HillClimbing, RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations, MaxRuntime

from tsp_as.classes import Solution
from tsp_as.destroy_operators import adjacent_destroy, random_destroy
from tsp_as.repair_operators import greedy_insert


def solve_alns(
    seed, init=None, data=None, max_runtime=None, max_iterations=None, **kwargs
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
        init = Solution(data, ordered)

    select = RouletteWheel([5, 2, 1, 0.5], 0.5, len(D_OPS), len(R_OPS))

    if max_runtime is not None:
        stop = MaxRuntime(max_runtime)
        accept = HillClimbing()
    else:
        assert max_iterations is not None
        stop = MaxIterations(max_iterations)
        accept = RecordToRecordTravel.autofit(init.objective(), 0.02, 0, max_iterations)

    res = alns.iterate(init, select, accept, stop, **kwargs)

    return res
