from tsp_as.classes import Solution


def increasing_scv(params, seed, **kwargs):
    """
    Creates a tour by ordering the clients in order of increasing SCV.
    """
    tour = params.service_scv.argsort().tolist()
    tour.remove(0)
    return Solution(params, tour)
