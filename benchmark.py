import argparse
import cProfile
import pstats
from datetime import datetime
from functools import partial
from glob import glob
from pathlib import Path

import numpy as np
import numpy.random as rnd
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.stop import MaxIterations
from alns.weights import SimpleWeights
from tqdm.contrib.concurrent import process_map

from tsp_as.classes import Params, Solution
from tsp_as.destroy_operators import random_destroy
from tsp_as.evaluations import heavy_traffic_optimal, heavy_traffic_pure, true_optimal
from tsp_as.repair_operators import greedy_insert


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=8)
    parser.add_argument("--instance_pattern", default="instances/atsp/*")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--profile", action="store_true")

    parser.add_argument("--objective", type=str, default="to")
    parser.add_argument("--n_destroy", type=int, default=2)
    parser.add_argument("--max_dim", type=int, default=10)

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
    alns.add_repair_operator(greedy_insert)

    init = Solution(params, np.arange(1, params.dimension).tolist())  # ordered
    weights = SimpleWeights([5, 2, 1, 0.5], 1, 1, 1)  # dummy scheme
    accept = RecordToRecordTravel.autofit(
        init_obj=init.objective(),
        start_gap=0.025,
        end_gap=0.0,
        num_iters=kwargs["max_iterations"],  # TODO this should work with time stop
    )
    stop = MaxIterations(kwargs["max_iterations"])

    res = alns.iterate(init, weights, accept, stop, **kwargs)
    stats = res.statistics

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

    if args.debug:  # disable parallelization to debug
        if args.profile:
            with cProfile.Profile() as profiler:
                data = [func(arg) for arg in func_args]

            stats = pstats.Stats(profiler).strip_dirs().sort_stats("time")
            stats.print_stats()

            now = datetime.now().isoformat()
            stats.dump_stats(f"tmp/profile-{now}.pstat")

        else:
            data = [func(arg) for arg in func_args]
    else:
        tqdm_kwargs = dict(max_workers=args.num_procs, unit="instance")
        data = process_map(func, func_args, **tqdm_kwargs)

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
