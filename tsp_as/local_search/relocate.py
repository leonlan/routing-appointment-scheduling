from tsp_as.classes import Solution


def relocate(solution: Solution, rng, **kwargs):
    """
    Improves the current solution in-place using the relocate neighborhood.
    A random customer is selected and optimally re-inserted. This continues
    until relocating none of the customers not lead to an improving move.
    """
    improved = True

    while improved:
        improved = False

        solution.update()
        current = solution.objective()

        for job in rng.choice(solution.tour, len(solution.tour), replace=False):
            solution.remove(job)
            opt_insert(solution, job)
            solution.update()

            if solution.objective() < current:
                improved = True
                break


def opt_insert(solution: Solution, cust: int):
    """
    Optimally insert the customer in the current tour.
    """
    idcs_costs = []

    for idx in range(len(solution.tour) + 1):
        cost = solution.insert_cost(idx, cust)
        idcs_costs.append((idx, cost))

    idx, _ = min(idcs_costs, key=lambda idx_cost: idx_cost[1])
    solution.insert(idx, cust)
