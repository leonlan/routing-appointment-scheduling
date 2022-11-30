from copy import deepcopy


def random_destroy(solution, rng, n_destroy, **kwargs):
    """
    Randomly remove clients from the solution.
    """
    destroyed = deepcopy(solution)

    for cust in rng.choice(destroyed.tour, n_destroy, replace=False):
        destroyed.unassigned.append(cust)
        destroyed.tour.remove(cust)

    destroyed.update()  # Update the costs

    assert len(solution.tour) == len(destroyed.tour) + n_destroy

    return destroyed
