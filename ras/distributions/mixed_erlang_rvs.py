import numpy as np
from numpy.random import Generator
from scipy.stats import gamma


def mixed_erlang_rvs(
    shapes: list[int],
    scales: list[float],
    weights: list[float],
    num_samples: int,
    rng: Generator,
) -> list[float]:
    """
    Generates samples from a mixed Erlang distribution with two phases.

    Parameters
    ----------
    shapes
        List of shape (phase) parameters for each Erlang distribution.
    scales
        List of scale parameters for each Erlang distribution.
        This can be computed by mean/phase.
    weights
        List of weights (probabilities) for each Erlang distribution.
    num_samples
        Number of samples to generate.
    rng
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Array of samples from the mixed Erlang distribution.
    """
    # Normalize weights to probabilities
    probs = np.array(weights) / sum(weights)

    # Select component Erlang distributions based on weights
    choices = rng.choice(len(probs), p=probs, size=num_samples)

    # Generate samples from the selected Erlang distributions
    samples = []
    for idx in np.unique(choices):
        num_choice = np.sum(choices == idx)
        samples.extend(gamma.rvs(shapes[idx], scale=scales[idx], size=num_choice))

    return samples
