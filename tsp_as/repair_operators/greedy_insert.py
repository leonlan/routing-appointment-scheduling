from tsp_as.classes import Solution


def greedy_insert(solution: Solution, rng, **kwargs):
    """
    Insert the unassigned customers into the best place, one-by-one.
    """
    rng.shuffle(solution.unassigned)

    while solution.unassigned:
        customer = solution.unassigned.pop()
        opt_insert(solution, customer)

    return solution


def opt_insert(solution: Solution, customer: int):
    """
    Optimally inserts the customer in the current visits.
    """
    idcs_costs = []

    for idx in range(len(solution.visits) + 1):
        cost = solution.insert_cost(idx, customer)
        idcs_costs.append((idx, cost))

    idx, _ = min(idcs_costs, key=lambda idx_cost: idx_cost[1])
    solution.insert(idx, customer)
