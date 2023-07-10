import argparse
import os

# BUG This is to avoid OpenBLAS from using multiple threads, see
# https://github.com/leonlan/tsp-as/issues/49
# https://github.com/numpy/numpy/issues/22928
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


from functools import partial
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from tsp_as import (
    double_orientation_tsp,
    full_enumeration,
    large_neighborhood_search,
    modified_tsp,
    nearest_neighbor_smallest_variance_first,
    smallest_variance_first,
    tsp,
)
from tsp_as.appointment.true_optimal import compute_idle_wait as true_objective_function
from tsp_as.classes import CostEvaluator, ProblemData
from tsp_as.plot import plot_graph


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("instances", nargs="+", help="Instance paths.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=8)
    parser.add_argument("--num_procs_enum", type=int, default=1)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="lns",
        choices=["lns", "tsp", "dotsp", "mtsp", "svf", "nnsvf", "enum"],
    )

    # Weight parameters for the cost function. Travel and idle time weightes
    # are deterministic, while the wait time weight is randomly generated
    # based on the seed.
    parser.add_argument("--weight_travel", type=float, default=1)
    parser.add_argument("--weight_idle", type=float, default=2.5)
    parser.add_argument("--weight_wait", type=int, default=10)

    parser.add_argument("--distance_scaling", type=float, default=1.0)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max_runtime", type=float)
    group.add_argument("--max_iterations", type=int)

    parser.add_argument("--sol_dir", type=str)
    parser.add_argument("--plot_dir", type=str)
    parser.add_argument("--breakdown", action="store_true")

    return parser.parse_args()


def maybe_mkdir(where: str):
    if where:
        directory = Path(where)
        directory.mkdir(parents=True, exist_ok=True)


def tabulate(headers, rows) -> str:
    # These lengths are used to space each column properly.
    lengths = [len(header) for header in headers]

    for row in rows:
        for idx, cell in enumerate(row):
            lengths[idx] = max(lengths[idx], len(str(cell)))

    header = [
        "  ".join(f"{h:<{l}s}" for l, h in zip(lengths, headers)),
        "  ".join("-" * l for l in lengths),
    ]

    content = [
        "  ".join(f"{str(c):>{l}s}" for l, c in zip(lengths, row)) for row in rows
    ]

    return "\n".join(header + content)


def make_cost_evaluator(
    data: ProblemData,
    weight_travel: float,
    weight_idle: float,
    weight_wait: int,
    seed: int,
) -> CostEvaluator:
    """
    Returns a cost evaluator based on the given weights. The customer-dependent
    wait time weight is randomly generated based on the given seed.
    """
    rng = np.random.default_rng(seed)

    obj_func = true_objective_function
    weights_wait = rng.integers(weight_wait, size=data.dimension) + 1

    return CostEvaluator(obj_func, weight_travel, weight_idle, weights_wait)


def cost_breakdown(data, cost_evaluator, solution):
    """
    Breakdown the cost of the solution.
    """
    tour_arcs = [0] + solution.visits, solution.visits + [0]
    dists = data.distances[tour_arcs]
    idle_times, wait_times = cost_evaluator.idle_wait_function(
        data, solution.visits, solution.schedule
    )

    headers = [
        "from",
        "to",
        "IA",
        "mean",
        "scv",
        "var",
        "dist",
        "idle",
        "wait",
        "weight_wait",
    ]
    rows = []
    locs = [0] + solution.visits + [0]
    n_locs = len(locs) - 1
    last = n_locs - 1

    cost_dist = 0
    cost_idle = 0
    cost_wait = 0
    for idx in range(n_locs):
        fr = locs[idx]
        to = locs[idx + 1]
        appointment = solution.schedule[idx] if idx < last else 0  # do not count last
        mean = data.arcs_mean[fr, to]
        scv = data.arcs_scv[fr, to]
        var = data.arcs_var[fr, to]
        dist = dists[idx]
        idle = idle_times[idx] if idx < last else 0  # do not count last
        wait = wait_times[idx] if idx < last else 0  # do not count last
        weight = cost_evaluator.wait_weights[to]

        row = (
            fr,
            to,
            round(appointment, 2),
            round(mean, 2),
            round(scv, 2),
            round(var, 2),
            round(dist, 2),
            round(idle, 2),
            round(wait, 2),
            round(weight, 2),
        )
        rows.append(row)

        cost_dist += dist * cost_evaluator.travel_weight
        cost_idle += idle * cost_evaluator.idle_weight
        cost_wait += wait * weight

    print(f"{cost_dist=:.2f}, {cost_idle=:.2f}, {cost_wait=:.2f}")

    return headers, rows


def solve(
    loc: str,
    seed: int,
    algorithm: str,
    weight_travel: float,
    weight_idle: float,
    weight_wait: int,
    distance_scaling: float,
    sol_dir: Optional[str],
    plot_dir: Optional[str],
    breakdown: Optional[bool],
    **kwargs,
):
    """
    Solves the instance using the ALNS package.
    """
    path = Path(loc)
    data = ProblemData.from_file(loc, distance_scaling=distance_scaling)

    # The cost seed is used to generate the customer-dependent wait times.
    # This seed is derived from the instance name.
    cost_seed = int(data.name.split("-")[1].strip("idx"))
    cost_evaluator = make_cost_evaluator(
        data, weight_travel, weight_idle, weight_wait, cost_seed
    )

    if algorithm == "lns":
        result = large_neighborhood_search(seed, data, cost_evaluator, **kwargs)
    elif algorithm == "tsp":
        result = tsp(seed, data, cost_evaluator, **kwargs)
    elif algorithm == "dotsp":
        result = double_orientation_tsp(seed, data, cost_evaluator, **kwargs)
    elif algorithm == "mtsp":
        result = modified_tsp(seed, data, cost_evaluator, **kwargs)
    elif algorithm == "svf":
        result = smallest_variance_first(seed, data, cost_evaluator)
    elif algorithm == "nnsvf":
        result = nearest_neighbor_smallest_variance_first(seed, data, cost_evaluator)
    elif algorithm == "enum":
        result = full_enumeration(seed, data, cost_evaluator, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

    best = result.solution

    if breakdown:
        print(tabulate(*cost_breakdown(data, cost_evaluator, best)))

    name = (
        path.stem
        + "-"
        + algorithm
        + f"travel{weight_travel}-idle{weight_idle}-wait{weight_wait}"
    )

    if sol_dir:
        where = Path(sol_dir) / (f"{name}" + ".sol")

        with open(where, "w") as fh:
            fh.write(str(best))

    if plot_dir:
        _, ax = plt.subplots(1, 1, figsize=[12, 12])
        plot_graph(ax, data, solution=best)
        where = Path(plot_dir) / (f"{name}-{algorithm}" + ".pdf")
        plt.savefig(where)

    cost_profile = str((weight_travel, weight_idle, weight_wait))
    return (
        path.stem,
        round(best.objective(), 3),
        result.iterations,
        round(result.runtime, 3),
        algorithm,
        cost_profile,
    )


def benchmark(instances: List[str], **kwargs):
    """
    Solves a list of instances, and prints a table with the results. Any
    additional keyword arguments are passed to ``solve()``.

    Parameters
    ----------
    instances
        Paths to the instances to solve.
    """
    maybe_mkdir(kwargs.get("sol_dir", ""))
    maybe_mkdir(kwargs.get("plot_dir", ""))

    if len(instances) == 1:
        res = solve(instances[0], **kwargs)
        print(res)
        return

    func = partial(solve, **kwargs)
    func_args = sorted(instances)

    tqdm_kwargs = {"max_workers": kwargs.get("num_procs", 1), "unit": "instance"}
    data = process_map(func, func_args, **tqdm_kwargs)

    dtypes = [
        ("inst", "U37"),
        ("obj", float),
        ("iters", int),
        ("time", float),
        ("alg", "U37"),
        ("cost-profile", "U37"),
    ]

    data = np.asarray(data, dtype=dtypes)
    headers = [
        "Instance",
        "Obj.",
        "Iters. (#)",
        "Time (s)",
        "Algorithm",
        "Cost profile",
    ]

    print("\n", tabulate(headers, data), "\n", sep="")
    print(f"      Avg. objective: {data['obj'].mean():.0f}")
    print(f"     Avg. iterations: {data['iters'].mean():.0f}")
    print(f"   Avg. run-time (s): {data['time'].mean():.2f}")


def main():
    benchmark(**vars(parse_args()))


if __name__ == "__main__":
    main()
