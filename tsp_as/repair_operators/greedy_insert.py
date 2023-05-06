def greedy_insert(solution, rng, **kwargs):
    """
    Insert the unassigned customers into the best place, one-by-one.
    """
    rng.shuffle(solution.unassigned)

    while solution.unassigned:
        cust = solution.unassigned.pop()
        best_cost, best_idx = None, None

        for idx in range(len(solution) + 1):
            cost = solution.insert_cost(idx, cust)

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_idx = idx

        # Inserting customers also updates the solution's cost
        solution.insert(best_idx, cust)

    return solution
