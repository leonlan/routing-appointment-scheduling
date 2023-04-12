from tsp_as.classes import Solution


def increasing_scv(data, seed, **kwargs):
    """
    Creates a tour by ordering the clients in order of increasing SCVs from
    the combined travel and service times.
    """
    tour = data.scvs.argsort().tolist()
    tour.remove(0)  # ignore the depot
    return Solution(data, tour)
