def get_means_scvs(tour, params):
    fr = [0] + tour
    to = tour + [0]

    means = params.means[fr, to]
    SCVs = params.scvs[fr, to]

    return means, SCVs


def get_alphas_transitions(tour, params):
    fr = [0] + tour
    to = tour + [0]

    alphas = tuple(params.alphas[fr, to])
    transition_matrices = tuple(params.transitions[fr, to])
    return alphas, transition_matrices
