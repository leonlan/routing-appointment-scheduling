import argparse
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.stop import MaxIterations
from alns.weights import SimpleWeights

from tsp_as.classes import Params, Solution
from tsp_as.evaluations import (heavy_traffic_optimal, heavy_traffic_pure,
                                true_optimal)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=8)
    parser.add_argument("--instance_pattern", default="instances/*")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max_iterations", type=int)

    return parser.parse_args()


def random_destroy(solution, rng, **kwargs):
    """
    Randomly remove clients from the solution.
    """
    destroyed = deepcopy(solution)
    n_destroy = kwargs["n_destroy"]

    for cust in rng.choice(destroyed.tour, n_destroy, replace=False):
        destroyed.unassigned.append(cust)
        destroyed.tour.remove(cust)

    assert len(solution.tour) == len(destroyed.tour) + n_destroy

    return destroyed


def greedy_insert(solution, rng, **kwargs):
    """
    Insert the unassigned customers into the best place, one-by-one.
    """
    rng.shuffle(solution.unassigned)

    while solution.unassigned:
        cust = solution.unassigned.pop()
        best_cost, best_idx = None, None

        # Find best insertion idx
        for idx in range(len(solution) + 1):
            cost = solution.insert_cost(idx, cust)

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_idx = idx

        solution.insert(best_idx, cust)

    return solution


def solve_alns(loc: str, seed: int = 1, **kwargs):
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
        start_gap=0.05,
        end_gap=0.0,
        num_iters=kwargs["max_iterations"],
    )

    stop = MaxIterations(kwargs["max_iterations"])

    res = alns.iterate(init, weights, accept, stop, **kwargs)
    stats = res.statistics

    return (
        path.stem,
        res.best_state.objective(),
        len(stats.objectives),
        round(stats.total_runtime, 3),
        params,
    )


def plot_trajectory(params, title):
    solutions = params.trajectory
    htp = [heavy_traffic_pure(sol, params)[1] for sol in solutions]
    hto = [heavy_traffic_optimal(sol, params)[1] for sol in solutions]
    to = [
        (true_optimal(sol, params)[1] if idx % 1 == 0 else None)
        for idx, sol in enumerate(solutions)
    ]

    fig = plt.figure(figsize=(20, 8))
    plt.plot(htp, marker="x", label="$\\mathscr{L}_{ht}(i, x^{ht})$")
    plt.plot(hto, marker="o", label="$\\mathscr{L}(i, x^{ht})$")
    plt.plot(to, marker="s", label="$\\mathscr{L}(i, x^*)$")

    plt.title(title)
    plt.ylabel("Idle + waiting")
    plt.ylim(min([x for x in to if x is not None]) * 0.9, min(hto) * 1.5)
    plt.xlabel("Iterations (#)")

    plt.grid(color="grey", linestyle="--", linewidth=0.25)
    plt.legend()

    plt.savefig(f"{title}.png")


def main():
    # Setup configuration
    config = {"max_iterations": 10, "n_destroy": 3, "max_dim": 10}
    solve = lambda path, objective: solve_alns(path, objective=objective, **config)

    # Solve the instance with provided objective function strategy
    path = Path("instances/atsp/p43.atsp")
    *_, params = solve(path, "htp")

    # Save figure with passed-in title
    plot_trajectory(params, f"Search trajectory guided by HTP\n Instance {path.stem}")


if __name__ == "__main__":
    main()
