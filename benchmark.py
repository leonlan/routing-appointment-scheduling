import argparse
import cProfile
import pstats
from datetime import datetime
from functools import partial
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from alns import ALNS
from alns.accept import RecordToRecordTravel, SimulatedAnnealing
from alns.stop import MaxIterations
from alns.weights import SimpleWeights
from tqdm.contrib.concurrent import process_map

from tsp_as.classes import Params, Solution
from tsp_as.destroy_operators import adjacent_destroy, random_destroy
from tsp_as.plot import plot_instance
from tsp_as.repair_operators import greedy_insert


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=8)
    parser.add_argument("--instance_pattern", default="instances/*")
    parser.add_argument("--profile", action="store_true")

    parser.add_argument("--objective", type=str, default="hto")
    parser.add_argument("--n_destroy", type=int, default=3)
    parser.add_argument("--max_dim", type=int, default=20)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max_runtime", type=float)
    group.add_argument("--max_iterations", type=int)

    return parser.parse_args()


def solve_alns(loc: str, seed: int, **kwargs):
    """
    Solver using ALNS package.
    """
    path = Path(loc)
    rng = rnd.default_rng(seed)

    params = Params.from_tsplib(
        path, rng, max_dim=kwargs["max_dim"], objective=kwargs["objective"]
    )

    alns = ALNS(rng)
    alns.add_destroy_operator(random_destroy)
    alns.add_destroy_operator(adjacent_destroy)
    alns.add_repair_operator(greedy_insert)

    init = Solution(params, np.arange(1, params.dimension).tolist())  # ordered
    weights = SimpleWeights([5, 2, 1, 0.5], 1, 1, 0.8)  # dummy scheme
    accept = SimulatedAnnealing.autofit(
        init_obj=init.objective(),
        worse=0.025,
        accept_prob=0.5,
        num_iters=kwargs["max_iterations"],  # TODO this should work with time stop
    )
    stop = MaxIterations(kwargs["max_iterations"])

    res = alns.iterate(init, weights, accept, stop, **kwargs)
    stats = res.statistics

    # Plot the solution if coords are available
    if np.any(params.coords):
        fig, ax = plt.subplots(figsize=[10, 7.5], dpi=150)
        plot_instance(ax, params, res.best_state)
        fig.savefig(f"tmp/{path.stem}")
        plt.close()

    return (
        path.stem,
        res.best_state.objective(),
        len(stats.objectives),
        round(stats.total_runtime, 3),
    )


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


def main():
    args = parse_args()

    func = partial(solve_alns, **vars(args))
    func_args = sorted(glob(args.instance_pattern))

    if args.num_procs > 1:
        tqdm_kwargs = dict(max_workers=args.num_procs, unit="instance")
        data = process_map(func, func_args, **tqdm_kwargs)
    else:  # process_map cannot be used with interactive debugging
        if args.profile:
            with cProfile.Profile() as profiler:
                data = [func(args) for args in func_args]

            stats = pstats.Stats(profiler).strip_dirs().sort_stats("time")
            stats.print_stats()

            now = datetime.now().isoformat()
            stats.dump_stats(f"tmp/profile-{now}.pstat")
        else:
            data = [func(args) for args in func_args]

    dtypes = [
        ("inst", "U37"),
        ("obj", int),
        ("iters", int),
        ("time", float),
    ]

    data = np.asarray(data, dtype=dtypes)

    headers = [
        "Instance",
        "Objective",
        "Iters. (#)",
        "Time (s)",
    ]

    table = tabulate(headers, data)

    print("\n", table, "\n", sep="")

    obj_all = data["obj"]

    print(f"      Avg. objective: {obj_all.mean():.0f}")
    print(f"     Avg. iterations: {data['iters'].mean():.0f}")
    print(f"   Avg. run-time (s): {data['time'].mean():.2f}")


if __name__ == "__main__":
    main()
