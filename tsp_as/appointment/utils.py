def get_means_scvs(visits, data):
    """
    Returns the means and SCVs for all legs in the visits expect the last one.
    """
    frm, to = visits2from_to(visits)

    means = data.means[frm, to]
    SCVs = data.scvs[frm, to]

    return means, SCVs


def get_vars(visits, data):
    frm, to = visits2from_to(visits)

    return data.vars[frm, to]


def get_leg_data(visits, data):
    """
    Returns the means and SCVs for all legs in the visits expect the last one.
    """
    frm, to = visits2from_to(visits)

    means = data.means[frm, to]
    SCVs = data.scvs[frm, to]
    variances = data.vars[frm, to]

    return means, SCVs, variances


def get_alphas_transitions(visits, data):
    frm, to = visits2from_to(visits)

    alphas = tuple(data.alphas[frm, to])
    transition_matrices = tuple(data.transitions[frm, to])
    return alphas, transition_matrices


def visits2from_to(visits):
    frm = [0] + visits[:-1]
    to = visits
    return frm, to
