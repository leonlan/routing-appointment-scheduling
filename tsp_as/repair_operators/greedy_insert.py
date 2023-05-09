def greedy_insert(solution, rng, **kwargs):
    """
    Insert the unassigned customers into the best place, one-by-one.
    """
    rng.shuffle(solution.unassigned)

    while solution.unassigned:
        customer = solution.unassigned.pop()
        solution.opt_insert(customer)

    return solution
