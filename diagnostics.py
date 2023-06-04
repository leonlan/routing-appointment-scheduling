def cost_breakdown(data, solution):
    """
    Breakdown the cost of the solution.
    """
    tour_arcs = [0] + solution.visits, solution.visits + [0]
    dists = data.distances[tour_arcs]
    idle_times, wait_times = solution.cost_evaluator.idle_wait_function(
        solution.visits, solution.schedule, solution.data
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
        weight = solution.cost_evaluator.wait_weights[to]

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

        cost_dist += dist * solution.cost_evaluator.travel_weight
        cost_idle += idle * solution.cost_evaluator.idle_weight
        cost_wait += wait * weight

    print(f"{cost_dist=:.2f}, {cost_idle=:.2f}, {cost_wait=:.2f}")

    return headers, rows


def tabulate(headers, rows) -> str:  # noqa
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
