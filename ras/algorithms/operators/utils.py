from ras.classes import ProblemData, Solution


def compute_unassigned(data: ProblemData, solution: Solution):
    """
    Returns the unassigned clients in the given solution.
    """
    assigned = {client for route in solution.routes for client in route}
    return [client for client in range(1, data.dimension) if client not in assigned]
