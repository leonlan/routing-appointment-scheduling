def get_means_scvs(tour, data):
    fr = [0] + tour
    to = tour + [0]

    means = data.means[fr, to]
    SCVs = data.scvs[fr, to]

    return means, SCVs


def get_alphas_transitions(tour, data):
    fr = [0] + tour
    to = tour + [0]

    alphas = tuple(data.alphas[fr, to])
    transition_matrices = tuple(data.transitions[fr, to])
    return alphas, transition_matrices
