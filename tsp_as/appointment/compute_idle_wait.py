import tsp_as.appointment.heavy_traffic as ht
import tsp_as.appointment.true_optimal as to


def compute_idle_wait(
    visits, data, cost_evaluator
) -> tuple[list[float], list[float], list[float]]:
    """
    Computes the idle and waiting times.

    Returns
    -------
    list[float]
        The inter-appointment times.
    list[float]
        The idle times at each client visit.
    list[float]
        The waiting times at each client visit.
    """
    if data.objective in ["htp", "hto"]:
        schedule = ht.compute_schedule(visits, data, cost_evaluator)

        if data.objective == "htp":
            idle, wait = ht.compute_objective(visits, schedule, data)
        elif data.objective == "hto":
            idle, wait = to.compute_idle_wait(visits, schedule, data)

        return schedule, idle, wait

    elif data.objective == "to":
        schedule, idle, wait = to.compute_schedule_and_idle_wait(
            visits, data, cost_evaluator
        )
        return schedule, idle, wait
