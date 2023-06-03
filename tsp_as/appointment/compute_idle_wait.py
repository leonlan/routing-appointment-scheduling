import tsp_as.appointment.heavy_traffic as ht
import tsp_as.appointment.lag as lag
import tsp_as.appointment.true_optimal as to


def compute_idle_wait(tour, data) -> tuple[list[float], list[float], list[float]]:
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
    if data.objective in ["htp", "hto", "htl"]:
        schedule = ht.compute_schedule(tour, data)

        if data.objective == "htp":
            # FIXME Separate idle and wait costs in heavy traffic, low prio because not used
            return schedule, ht.compute_objective(tour, data)
        elif data.objective == "hto":
            idle, wait = to.compute_idle_wait(tour, schedule, data)
            return schedule, idle, wait
        elif data.objective == "htl":
            idle, wait = lag.compute_idle_wait(tour, schedule, data)
            return schedule, idle, wait

    if data.objective == "to":
        schedule, idle, wait = to.compute_schedule_and_idle_wait(tour, data)
        return schedule, idle, wait

    raise ValueError(f"{data.objective=} unknown.")
