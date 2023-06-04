import argparse
import os

# BUG This is to avoid OpenBLAS from using multiple threads, see
# https://github.com/leonlan/tsp-as/issues/49
# https://github.com/numpy/numpy/issues/22928
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from diagnostics import cost_breakdown
from tsp_as import (
    full_enumeration,
    increasing_variance,
    solve_alns,
    solve_modified_tsp,
    solve_tsp,
)
from tsp_as.classes import CostEvaluator, ProblemData, Solution
from tsp_as.plot import plot_graph


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("instances", nargs="+", help="Instance paths.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=8)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="alns",
        choices=["alns", "tsp", "mtsp", "scv", "var", "enum"],
    )

    parser.add_argument("--objective", type=str, default="hto")
    parser.add_argument("--final_objective", type=str, default="to")
    parser.add_argument(
        "--cost_profile",
        type=str,
        default="small",
        choices=[
            "small",  # (0.1, 0.3, 0.6)
            "medium",  # (0.2, 0.25, 0.55)
            "large",  # (0.3, 0.2, 0.5)
        ],
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max_runtime", type=float)
    group.add_argument("--max_iterations", type=int)

    parser.add_argument("--sol_dir", type=str)
    parser.add_argument("--plot_dir", type=str)
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


def make_cost_evaluator(data: ProblemData, cost_profile: str, seed) -> CostEvaluator:
    """
    Returns a cost evaluator based on the given cost profile.
    """
    rng = np.random.default_rng(seed)

    def generate_weights(mean_weight):
        """Generates random weights around the given mean for each node."""
        return 2 * mean_weight * rng.random(data.dimension)

    if cost_profile == "small":
        return CostEvaluator(0.1, 0.3, generate_weights(0.6))
    elif cost_profile == "medium":
        return CostEvaluator(0.2, 0.25, generate_weights(0.55))
    elif cost_profile == "large":
        return CostEvaluator(0.3, 0.2, generate_weights(0.5))
    else:
        raise ValueError(f"Unknown cost profile {cost_profile}")


def solve(
    loc: str,
    seed: int,
    algorithm: str,
    final_objective: str,
    cost_profile: str,
    sol_dir: Optional[str],
    plot_dir: Optional[str],
    **kwargs,
):
    """
    Solves the instance using the ALNS package.
    """
    path = Path(loc)
    data = ProblemData.from_file(loc, **kwargs)
    cost_evaluator = make_cost_evaluator(data, cost_profile, seed)

    if algorithm == "alns":
        res = solve_alns(seed, data, cost_evaluator, **kwargs)
    elif algorithm == "tsp":
        res = solve_tsp(seed, data, cost_evaluator, **kwargs)
    elif algorithm == "mtsp":
        res = solve_modified_tsp(seed, data, cost_evaluator, **kwargs)
    elif algorithm == "var":
        res = increasing_variance(seed, data, cost_evaluator)
    elif algorithm == "enum":
        res = full_enumeration(seed, data, cost_evaluator)

    # Final evaluation of the solution based on another objective function
    final_data = deepcopy(data)
    final_data.objective = final_objective
    best = Solution(final_data, cost_evaluator, res.best_state.visits)
    print(tabulate(*cost_breakdown(best)))

    if sol_dir:
        instance_name = Path(loc).stem
        where = Path(sol_dir) / (f"{instance_name}-{algorithm}" + ".sol")

        with open(where, "w") as fh:
            fh.write(str(res.best_state))

    if plot_dir:
        _, ax = plt.subplots(1, 1, figsize=[12, 12])
        plot_graph(ax, data, solution=best)
        instance_name = Path(loc).stem
        where = Path(plot_dir) / (
            f"{instance_name}-{algorithm}-{final_objective}" + ".pdf"
        )
        plt.savefig(where)

    return (
        path.stem,
        best.cost,
        len(res.statistics.objectives),
        round(res.statistics.total_runtime, 3),
        algorithm,
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
        ("obj", int),
        ("iters", int),
        ("time", float),
        ("alg", "U37"),
    ]

    data = np.asarray(data, dtype=dtypes)
    headers = ["Instance", "Obj.", "Iters. (#)", "Time (s)", "Algorithm"]

    print("\n", tabulate(headers, data), "\n", sep="")
    print(f"      Avg. objective: {data['obj'].mean():.0f}")
    print(f"     Avg. iterations: {data['iters'].mean():.0f}")
    print(f"   Avg. run-time (s): {data['time'].mean():.2f}")


def main():
    benchmark(**vars(parse_args()))


if __name__ == "__main__":
    main()
