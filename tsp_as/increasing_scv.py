from tsp_as.classes import Solution


def increasing_scv(data, seed, **kwargs):
    """
    Creates a tour by ordering the clients in order of increasing SCV.
    """
    tour = data.service_scv.argsort().tolist()
    tour.remove(0)  # Ignore the depot
    return Solution(data, tour)
