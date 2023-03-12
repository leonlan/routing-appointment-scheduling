def solve(loc: str, seed: int, **kwargs):
    """
    Solve the instance.
    """
    path = Path(loc)
    rng = rnd.default_rng(seed)

    params = Params.from_tsplib(path, rng, **kwargs)

    alns = ALNS(rng)
    alns.add_destroy_operator(random_destroy)
    alns.add_destroy_operator(adjacent_destroy)
    alns.add_repair_operator(greedy_insert)

    init = Solution(params, np.arange(1, params.dimension).tolist())  # ordered
    weights = SimpleWeights([5, 2, 1, 0.5], 2, 2, 0.8)
    accept = HillClimbing()
    stop = MaxIterations(kwargs["max_iterations"])
    # stop = MaxRuntime(kwargs["max_runtime"] * 60)

    res = alns.iterate(init, weights, accept, stop, **kwargs)
    stats = res.statistics

    # # Compute the final, optimal objective
    # schedule, cost = res.best_state.compute_optimal_schedule()
    # res.best_state.schedule = schedule

    if np.any(params.coords):
        fig, ax = plt.subplots(figsize=[16, 12], dpi=150)
        plot_graph(ax, params, res.best_state)
        fig.savefig(f"tmp/{path.stem}-dim{kwargs['max_dim']}.svg")
        plt.close()

    return (
        path.stem,
        res.best_state.objective(),
        # cost,
        len(stats.objectives),
        round(stats.total_runtime, 3),
    )


def main():
    # Specify instance characteristics
    coordinates = ...
    distances = ...
    distances_scv_min = ...
    distances_scv_max = ...

    service_scv_min = ...
    service_scv_max = ...

    omega_travel = 0.2
    omega_idle = 0.2
    omega_wait = 0.6

    objective = "hto"  # hto, htt, to

    # Setup algorithm to solve the instance

    # Plot results and show the output

    pass


if __name__ == "__main__":
    main()
