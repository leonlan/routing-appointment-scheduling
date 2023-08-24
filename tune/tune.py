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

import numpy as np
from tqdm.contrib.concurrent import process_map

from tsp_as import large_neighborhood_search
from tsp_as.appointment.true_optimal import compute_idle_wait as true_objective_function
from tsp_as.classes import CostEvaluator, ProblemData


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("instances", nargs="+", help="Instance paths.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=8)
    parser.add_argument("--algorithm", type=str, default="lns")

    parser.add_argument("--weight_travel", type=float, default=1)
    parser.add_argument("--weight_idle", type=float, default=2.5)
    parser.add_argument("--weight_wait", type=int, default=10)

    parser.add_argument("--max_num_destroy", type=int, default=6)
    parser.add_argument("--rrt_start_threshold_pct", type=float, default=0.05)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max_runtime", type=float)
    group.add_argument("--max_iterations", type=int)
    group.add_argument("--benchmark_runtime", action="store_true")

    return parser.parse_args()


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


def solve(
    loc: str,
    seed: int,
    algorithm: str,
    weight_travel: float,
    weight_idle: float,
    weight_wait: int,
    max_num_destroy: int,
    rrt_start_threshold_pct: float,
    benchmark_runtime: Optional[bool],
    **kwargs,
):
    """
    Solves the instance using the ALNS package.
    """
    path = Path(loc)
    data = ProblemData.from_file(loc)

    # The cost seed is used to generate the customer-dependent wait times.
    # This seed is derived from the instance name.
    cost_seed = int(data.name.split("-")[1].strip("idx"))
    cost_evaluator = make_cost_evaluator(
        data, weight_travel, weight_idle, weight_wait, cost_seed
    )

    if benchmark_runtime:
        runtimes_by_size = {
            6: 5,
            8: 10,
            10: 15,
            15: 30,
            20: 60,
            25: 120,
            30: 180,
            35: 240,
            40: 300,
        }
        kwargs["max_runtime"] = runtimes_by_size[data.dimension - 1]

    result = large_neighborhood_search(
        seed,
        data,
        cost_evaluator,
        max_num_destroy=max_num_destroy,
        rrt_start_threshold_pct=rrt_start_threshold_pct,
        **kwargs,
    )

    best = result.solution
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
