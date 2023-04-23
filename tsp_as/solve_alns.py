import numpy as np
import numpy.random as rnd
from alns import ALNS
from alns.accept import HillClimbing
from alns.stop import MaxIterations, MaxRuntime
from alns.weights import SimpleWeights

from tsp_as.classes import Solution
from tsp_as.destroy_operators import adjacent_destroy, random_destroy
from tsp_as.repair_operators import greedy_insert


def solve_alns(
    seed, init=None, data=None, max_runtime=None, max_iterations=None, **kwargs
):
    rng = rnd.default_rng(seed)

    alns = ALNS(rng)
    alns.add_destroy_operator(random_destroy)
    alns.add_destroy_operator(adjacent_destroy)
    alns.add_repair_operator(greedy_insert)

    if init is None:
        ordered = np.arange(1, data.dimension).tolist()
        init = Solution(data, ordered)

    weights = SimpleWeights([5, 2, 1, 0.5], 2, 2, 0.8)
    accept = HillClimbing()

    if max_runtime is not None:
        stop = MaxRuntime(max_runtime)
    else:
        assert max_iterations is not None
        stop = MaxIterations(max_iterations)

    res = alns.iterate(init, weights, accept, stop, **kwargs)

    return res
