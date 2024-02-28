import time
from itertools import product

import gurobipy as gp
from gurobipy import GRB, quicksum
from numpy.random import default_rng

from ras.appointment.true_optimal import compute_optimal_schedule
from ras.classes import CostEvaluator, ProblemData, Solution
from ras.Result import Result

from .utils import _sample_distance_matrices, _sample_service_times


def tsang(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    max_runtime: int,
    num_scenarios: int,
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

    m = gp.Model()
    N = data.dimension
    S = num_scenarios

    rng = default_rng(seed)

    distance = _sample_distance_matrices(data, num_scenarios, rng)

    if num_scenarios == 1:
        distance[:, :, 0] = data.distances

    service = _sample_service_times(data, num_scenarios, rng)

    x = m.addVars(N, N, vtype=GRB.BINARY, name="order")
    a = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="appointment time")
    idle = m.addVars(N, S, vtype=GRB.CONTINUOUS, lb=0, name="idle")
    wait = m.addVars(N, S, vtype=GRB.CONTINUOUS, lb=0, name="wait")
    total_travel = m.addVars(S, vtype=GRB.CONTINUOUS, lb=0, name="total_travel")

    # Objective
    travel_weight = cost_evaluator.travel_weight
    idle_weight = cost_evaluator.idle_weight
    wait_weights = cost_evaluator.wait_weights

    avg_travel = sum(travel_weight * total_travel[s] for s in range(S)) / S
    avg_idle = sum(idle_weight * idle[i, s] for i in range(N) for s in range(S)) / S
    avg_wait = sum(wait_weights[i] * wait[i, s] for i in range(N) for s in range(S)) / S

    m.setObjective(avg_travel + avg_idle + avg_wait, GRB.MINIMIZE)

    # Each customer is assigned to exactly one position
    for i in range(1, N):
        m.addConstr(sum(x[i, j] for j in range(1, N)) == 1)

    # Each position is assign to exactly one customer
    for j in range(1, N):
        m.addConstr(sum(x[i, j] for i in range(1, N)) == 1)

    # Appointment times are ordered.
    for i in range(1, N):
        m.addConstr(a[i - 1] <= a[i])

    for s in range(S):
        lhs = wait[1, s] - idle[1, s]
        travel2first = quicksum(distance[0, i, s] * x[i, 1] for i in range(1, N))
        rhs = travel2first - a[1]
        m.addConstr(lhs == rhs)

    for j, s in product(range(2, N), range(S)):
        lhs = wait[j, s] - wait[j - 1, s] - idle[j, s]
        rhs = a[j - 1] - a[j]  # inter appointment time
        rhs += quicksum(service[i, s] * x[i, j - 1] for i in range(1, N))
        rhs += quicksum(
            quicksum(
                distance[i, k, s] * x[i, j - 1] * x[k, j] for k in range(1, N) if i != k
            )
            for i in range(1, N)
        )
        m.addConstr(lhs == rhs)

    # NOTE We skip the overtime constraint because we don't have overtime.

    for s in range(S):
        # Distances from client <-> client.
        expr1 = quicksum(
            distance[i, k, s] * x[i, j - 1] * x[k, j]
            for j in range(2, N)
            for i in range(1, N)
            for k in range(1, N)
            if i != k
        )

        # Distances from depot <-> client.
        expr2 = quicksum(
            distance[0, i, s] * x[i, 1] + distance[i, 0, s] * x[i, N - 1]
            for i in range(1, N)
        )

        m.addConstr(total_travel[s] == expr1 + expr2)

    m.setParam("TimeLimit", max_runtime)
    m.optimize()

    try:
        vals = m.getAttr("X", x)
    except gp.GurobiError:  # no feasible solution
        dummy = list(range(1, data.dimension))
        empty = Solution(data, cost_evaluator, dummy, dummy)
        empty.cost = -1
        return Result(empty, time.perf_counter() - start, 0)

    vals = m.getAttr("X", x)
    positions = [(i, j) for i, j in x.keys() if vals[i, j] == 1]
    order = sorted(positions, key=lambda x: x[1])
    visits = [x[0] for x in order]

    schedule = compute_optimal_schedule(data, cost_evaluator, visits)
    solution = Solution(data, cost_evaluator, visits, schedule)

    # Debugging
    # appt_times = {i: a[i].X for i in range(N)}
    # idle_vals = {(i, j): idle[i, j].X for i in range(N) for j in range(S)}
    # wait_vals = {(i, j): wait[i, j].X for i in range(N) for j in range(S)}
    # dists = data.distances[[0] + visits, visits + [0]] # use mean here, not samples
    # serv = service[[0] + visits, 0]  # use samples here, not the mean

    # expected = np.cumsum(dists + serv)
    # appt times should be exactly dist + service
    # breakpoint()

    return Result(solution, time.perf_counter() - start, 0)
