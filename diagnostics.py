from tsp_as.appointment.true_optimal import compute_idle_wait_per_client


def cost_breakdown(solution):
    """
    Breakdown the cost of the solution.
    """
    fr, to = [0] + solution.tour, solution.tour + [0]
    data = solution.data
    dists = data.distances[fr, to]
    idle_times, wait_times = compute_idle_wait_per_client(
        solution.tour, solution.schedule, solution.data
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
    ]
    rows = []
    locs = [0] + solution.tour + [0]
    n_locs = len(locs) - 1
    last = n_locs - 1

    total_dist = 0
    total_idle = 0
    total_wait = 0
    for idx in range(n_locs):
        fr = locs[idx]
        to = locs[idx + 1]
        appointment = solution.schedule[idx] if idx < last else 0  # do not count last
        mean = data.means[fr, to]
        scv = data.scvs[fr, to]
        var = data.vars[fr, to]
        dist = dists[idx]
        idle = idle_times[idx] if idx < last else 0  # do not count last
        wait = wait_times[idx] if idx < last else 0  # do not count last

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
        )
        rows.append(row)
        total_dist += dist
        total_idle += idle
        total_wait += wait

    print(f"{total_dist=:.2f}, {total_idle=:.2f}, {total_wait=:.2f}")

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
