import time

import gurobipy as gp
from gurobipy import GRB, quicksum
from numpy.random import default_rng

from ras.appointment.true_optimal import compute_optimal_schedule
from ras.classes import CostEvaluator, ProblemData, Result, Solution


def zhan_mip(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    max_runtime: int,
    num_scenarios: int,
    **kwargs,
):
    """
    Solves the routing and appointment scheduling problem using the sample
    average approximation method. The problem is solved by solving the
    extensive form of the problem using the Gurobi solver.

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
    big_M = 10000

    m = gp.Model()
    N = data.dimension
    S = num_scenarios

    rng = default_rng(seed)
    distances = _sample_distance_matrices(data, num_scenarios, rng)
    service = _sample_service_times(data, num_scenarios, rng)

    y = m.addVars(N, N, vtype=GRB.BINARY, name="order")
    x = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="appointment time")
    z = m.addVars(N, S, vtype=GRB.CONTINUOUS, lb=0, name="start service")

    idle = m.addVars(N, S, lb=0, vtype=GRB.CONTINUOUS, name="idle time")
    wait = m.addVars(N, S, lb=0, vtype=GRB.CONTINUOUS, name="service time")

    # Objective
    travel_weight = cost_evaluator.travel_weight
    idle_weight = cost_evaluator.idle_weight
    wait_weights = cost_evaluator.wait_weights

    total_travel = quicksum(
        travel_weight * data.distances[i, j] * y[i, j]
        for i in range(N)
        for j in range(N)
    )
    avg_total_idle = (
        quicksum(idle_weight * idle[i, s] for i in range(N) for s in range(S)) / S
    )
    avg_total_wait = (
        quicksum(wait_weights[i] * wait[i, s] for i in range(N) for s in range(S)) / S
    )

    m.setObjective(total_travel + avg_total_idle + avg_total_wait, GRB.MINIMIZE)

    # Exactly one edge leaving each node
    for i in range(N):
        m.addConstr(quicksum(y[i, j] for j in range(N)) == 1)

    # Exactly one edge entering each node
    for j in range(N):
        m.addConstr(quicksum(y[i, j] for i in range(N)) == 1)

    # No self-loops
    for i in range(N):
        m.addConstr(y[i, i] == 0)

    # Start of service is equal to the appointment time plus waiting time
    for i in range(N):
        for s in range(S):
            m.addConstr(z[i, s] == x[i] + wait[i, s])

    # Start of service at j is bounded below by start of service at i + service
    # + travel if there is an arc from i to j
    for i in range(N):
        for j in range(N):
            if j == 0:  # ignore when going back to the depot
                continue

            for s in range(S):
                m.addConstr(
                    z[j, s]
                    >= z[i, s]
                    + service[i, s]
                    + distances[i, j, s]
                    - big_M * (1 - y[i, j])
                )

    # The idle time is equal to the difference of z_j and z_i + service + travel,
    # given that arc (i, j) is selected.
    for i in range(N):
        for j in range(N):
            if j == 0:  # ignore when going back to the depot
                continue

            for s in range(S):
                m.addConstr(
                    idle[j, s]
                    >= z[j, s]
                    - z[i, s]
                    - service[i, s]
                    - distances[i, j, s]
                    - big_M * (1 - y[i, j]),
                )

    m.setParam("TimeLimit", max_runtime)
    m.optimize()

    return m, y, x, idle, wait


def saa(
    seed: int,
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    max_runtime: int,
    num_scenarios: int,
    **kwargs,
) -> Result:
    """
    Wrapper around ``zhan_mip`` to convert to RAS solution.
    """
    start = time.perf_counter()

    m, y, *_ = zhan_mip(
        seed, data, cost_evaluator, max_runtime, num_scenarios, **kwargs
    )

    try:
        vals = m.getAttr("X", y)
    except gp.GurobiError:  # no feasible solution
        dummy = list(range(1, data.dimension))
        empty = Solution(data, cost_evaluator, dummy, dummy)
        empty.cost = -1
        return Result(empty, time.perf_counter() - start, 0)

    vals = m.getAttr("X", y)
    edges = [(i, j) for i, j in y.keys() if vals[i, j] > 0.5]
    tour = find_shortest_cycle(edges)
    visits = tour[1:]  # ignore depot

    schedule = compute_optimal_schedule(data, cost_evaluator, visits)
    solution = Solution(data, cost_evaluator, visits, schedule)

    return Result(solution, time.perf_counter() - start, 0)


def find_shortest_cycle(edges: list[tuple[int, int]]) -> list[int]:
    """
    Finds the shortest cycle in the graph defined by the edges. It is assumed
    that the graph is connected and consists only of cycles.
    """
    cycle = None
    unvisited = {node for edge in edges for node in edge}

    neighbors: dict[int, set] = {i: set() for i in unvisited}
    for i, j in edges:
        neighbors[i].add(j)

    while unvisited:
        current = [unvisited.pop()]

        # Visit the next node if unvisited and add to the cycle
        while candidates := neighbors[current[-1]].intersection(unvisited):
            selected = candidates.pop()
            current.append(selected)
            unvisited.remove(selected)

        if cycle is None or len(current) < len(cycle):
            cycle = current

    assert cycle is not None
    return cycle


from itertools import product

import numpy as np
from numpy.random import Generator

from ras.classes import ProblemData
from ras.distributions import (
    fit_hyperexponential,
    fit_mixed_erlang,
    hyperexponential_rvs,
    mixed_erlang_rvs,
)


def _sample_distance_matrices(
    data: ProblemData, num_samples: int, rng: Generator
) -> np.ndarray:
    """
    Samples a number of distances matrix.

    Parameters
    ----------
    data
        ProblemData object.
    num_samples
        The number of samples.
    rng
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Sampled distances matrix.
    """
    distances = np.zeros((data.dimension, data.dimension, num_samples))

    for i, j in product(range(data.dimension), repeat=2):
        if i == j:
            continue

        mean, scv = data.distances[i, j], data.distances_scv[i, j]

        if scv < 1:  # Mixed Erlang case
            K, p, mu = fit_mixed_erlang(mean, scv)
            distances[i, j, :] = mixed_erlang_rvs(
                [K - 1, K], [1 / mu, 1 / mu], [p, (1 - p)], num_samples, rng
            )

        else:  # Hyperexponential case
            p, mu1, mu2 = fit_hyperexponential(mean, scv)
            distances[i, j, :] = hyperexponential_rvs(
                [1 / mu1, 1 / mu2], [p, (1 - p)], num_samples, rng
            )

    return distances


def _sample_service_times(
    data: ProblemData, num_samples: int, rng: Generator
) -> np.ndarray:
    """
    Samples a number of service times vectors.

    Parameters
    ----------
    data
        ProblemData object.
    num_samples
        The number of samples.
    rng
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Sampled service times.
    """
    service = np.zeros((data.dimension, num_samples))

    for i in range(data.dimension):
        if i == 0:
            continue

        mean, scv = data.service[i], data.service_scv[i]

        if scv < 1:  # Mixed Erlang case
            K, p, mu = fit_mixed_erlang(mean, scv)
            service[i, :] = mixed_erlang_rvs(
                [K - 1, K], [1 / mu, 1 / mu], [p, (1 - p)], num_samples, rng
            )

        else:  # Hyperexponential case
            p, mu1, mu2 = fit_hyperexponential(mean, scv)
            service[i, :] = hyperexponential_rvs(
                [1 / mu1, 1 / mu2], [p, (1 - p)], num_samples, rng
            )

    return service
