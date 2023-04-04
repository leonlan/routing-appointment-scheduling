import tsp_as.appointment.heavy_traffic as ht
from tsp_as.appointment.lag import compute_objective_given_schedule_breakdown


def cost_breakdown(solution, data):
    """
    Breakdown the cost of the solution.
    """
    interappointment_times = solution.schedule

    fr, to = [0] + solution.tour, solution.tour + [0]
    dists = data.distances[fr, to]

    schedule = ht.compute_schedule(solution.tour, data)
    idle_wait = compute_objective_given_schedule_breakdown(
        solution.tour, schedule, data
    )

    headers = ["from", "to", "interappt. time", "dist.", "idle & wait"]
    rows = []
    locs = [0] + solution.tour + [0]
    for idx in range(len(locs) - 1):
        fr, to = locs[idx], locs[idx + 1]
        row = (
            fr,
            to,
            round(interappointment_times[idx], 2),
            round(dists[idx], 2),
            round(idle_wait[idx], 2),
        )
        rows.append(row)

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
