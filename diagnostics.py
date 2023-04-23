import tsp_as.appointment.heavy_traffic as ht
from tsp_as.appointment.true_optimal import compute_objective_given_schedule_breakdown


def cost_breakdown(solution, data):
    """
    Breakdown the cost of the solution.
    """
    interappointment_times = solution.schedule

    fr, to = [0] + solution.tour, solution.tour + [0]
    dists = data.distances[fr, to] * data.omega_travel

    schedule = ht.compute_schedule(solution.tour, data)
    idle_wait = compute_objective_given_schedule_breakdown(
        solution.tour, schedule, data
    )

    headers = ["from", "to", "var", "mean", "IA", "dist", "idle & wait"]
    rows = []
    locs = [0] + solution.tour + [0]
    total = 0
    n_locs = len(locs) - 1
    last = n_locs - 1
    for idx in range(n_locs):
        fr, to = locs[idx], locs[idx + 1]

        x = round(interappointment_times[idx], 2)
        var = round(data.vars[fr, to], 2)
        mean = round(data.means[fr, to], 2)
        dist = round(dists[idx], 2)
        iw = round(idle_wait[idx], 2)
        iw_ = iw if idx < last else 0  # do not count last

        row = (
            fr,
            to,
            var,
            mean,
            x,
            dist,
            iw_,
        )
        rows.append(row)
        total += dist + iw_

        print((dist + iw) / x)

    print(total)
    return headers, rows


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
