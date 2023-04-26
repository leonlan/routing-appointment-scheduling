def get_means_scvs(tour, data):
    """
    Returns the means and SCVs for all legs in the tour expect the last one.
    """
    frm = [0] + tour[:-1]
    to = tour

    means = data.means[frm, to]
    SCVs = data.scvs[frm, to]

    return means, SCVs


def get_leg_data(tour, data):
    """
    Returns the means and SCVs for all legs in the tour expect the last one.
    """
    frm = [0] + tour[:-1]
    to = tour

    means = data.means[frm, to]
    SCVs = data.scvs[frm, to]
    variances = data.vars[frm, to]

    return means, SCVs, variances


def get_alphas_transitions(tour, data):
    frm = [0] + tour[:-1]
    to = tour

    alphas = tuple(data.alphas[frm, to])
    transition_matrices = tuple(data.transitions[frm, to])
    return alphas, transition_matrices
