import argparse
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from diagnostics import cost_breakdown
from tsp_as import (
    full_enumeration,
    increasing_scv,
    increasing_variance,
    solve_alns,
    solve_modified_tsp,
    solve_tsp,
)
from tsp_as.classes import ProblemData, Solution
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
    parser.add_argument("--final_objective", type=str, default="hto")
    parser.add_argument("--omega_travel", type=float, default=1 / 3)
    parser.add_argument("--omega_idle", type=float, default=1 / 3)
    parser.add_argument("--omega_wait", type=float, default=1 / 3)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max_runtime", type=float)
    group.add_argument("--max_iterations", type=int)

    parser.add_argument("--sol_dir", type=str, default=None)
    parser.add_argument("--plot_dir", type=str, default=None)
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


def solve(
    loc: str,
    seed: int,
    algorithm: str,
    final_objective: str,
    sol_dir: Optional[str],
    plot_dir: Optional[str],
    **kwargs,
):
    """
    Solves the instance using the ALNS package.
    """
    path = Path(loc)

    data = ProblemData.from_file(loc, **kwargs)

    if algorithm == "alns":
        res = solve_alns(seed, data=data, **kwargs)
    elif algorithm == "tsp":
        res = solve_tsp(seed, data=data, **kwargs)
    elif algorithm == "mtsp":
        res = solve_modified_tsp(seed, data=data, **kwargs)
    elif algorithm == "scv":
        res = increasing_scv(seed, data)
    elif algorithm == "var":
        res = increasing_variance(seed, data)
    elif algorithm == "enum":
        res = full_enumeration(seed, data)

    # Final evaluation of the solution based on the HTO objective
    final_data = deepcopy(data)
    final_data.objective = final_objective
    best = Solution(final_data, res.best_state.tour)
    print(tabulate(*cost_breakdown(best, final_data)))

    # all_sols = [Solution(final_data, list(tour)) for tour in permutations(range(1, 6))]
    # optimal = min(all_sols, key=lambda sol: sol.cost)
    # print("Optimal tour: ", optimal.tour, optimal.cost)

    if sol_dir is not None:
        instance_name = Path(loc).stem
        where = Path(sol_dir) / (f"{instance_name}-{algorithm}" + ".sol")

        with open(where, "w") as fh:
            fh.write(str(res.best_state))

    if plot_dir is not None:
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

    # func = partial(solve, **kwargs)
    # func_args = sorted(instances)

    # tqdm_kwargs = {"max_workers": kwargs.get("num_procs", 1), "unit": "instance"}
    # data = process_map(func, func_args, **tqdm_kwargs)

    dtypes = [
        ("inst", "U37"),
        ("obj", int),
        ("iters", int),
        ("time", float),
    ]

    data = np.asarray(data, dtype=dtypes)
    headers = ["Instance", "Obj.", "Iters. (#)", "Time (s)"]

    print("\n", tabulate(headers, data), "\n", sep="")
    print(f"      Avg. objective: {data['obj'].mean():.0f}")
    print(f"     Avg. iterations: {data['iters'].mean():.0f}")
    print(f"   Avg. run-time (s): {data['time'].mean():.2f}")


def main():
    benchmark(**vars(parse_args()))


if __name__ == "__main__":
    main()
