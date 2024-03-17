import time
from typing import Optional

import pyvrp
from pyvrp.stop import MaxRuntime

from ras.appointment.true_optimal import compute_optimal_schedule
from ras.classes import CostEvaluator, ProblemData, Result, Route, Solution


def cvrp(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    max_runtime: Optional[float] = None,
    max_iterations: Optional[int] = None,
    **kwargs,
) -> Result:
    """
    Solves the given problem instance as deterministic CVRP, using the
    mean travel times as distance.
    """
    start = time.perf_counter()

    model = pyvrp.Model()

    model.add_depot(x=data.coords[0, 0], y=data.coords[0, 1])
    model.add_vehicle_type(data.num_vehicles)

    for idx in range(1, data.dimension):
        model.add_client(x=data.coords[idx, 0], y=data.coords[idx, 1])

    for idx1, frm in enumerate(model.locations):
        for idx2, to in enumerate(model.locations):
            model.add_edge(frm, to, data.distances[idx1, idx2])

    if max_iterations is not None:
        stop = pyvrp.stop.MaxIterations(max_iterations)
    elif max_runtime is not None:
        stop = MaxRuntime(max_runtime)
    else:
        raise ValueError("Either max_iterations or max_runtime must be provided.")

    res = model.solve(seed=seed, stop=stop)

    routes = []
    for route in res.best.get_routes():
        clients = route.visits()
        schedule = compute_optimal_schedule(data, cost_evaluator, clients)
        routes.append(Route(data, cost_evaluator, clients, schedule))

    return Result(
        Solution(routes),
        time.perf_counter() - start,
        len(res.stats.runtimes),
    )
