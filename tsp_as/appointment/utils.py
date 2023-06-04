def get_means_scvs(tour, data):
    """
    Returns the means and SCVs for all legs in the tour expect the last one.
    """
    frm, to = tour2from_to(tour)

    means = data.means[frm, to]
    SCVs = data.scvs[frm, to]

    return means, SCVs


def get_vars(tour, data):
    frm, to = tour2from_to(tour)

    return data.vars[frm, to]


def get_leg_data(tour, data):
    """
    Returns the means and SCVs for all legs in the tour expect the last one.
    """
    frm, to = tour2from_to(tour)

    means = data.means[frm, to]
    SCVs = data.scvs[frm, to]
    variances = data.vars[frm, to]

    return means, SCVs, variances


def get_alphas_transitions(tour, data):
    frm, to = tour2from_to(tour)

    alphas = tuple(data.alphas[frm, to])
    transition_matrices = tuple(data.transitions[frm, to])
    return alphas, transition_matrices


def tour2from_to(tour):
    frm = [0] + tour[:-1]
    to = tour
    return frm, to
