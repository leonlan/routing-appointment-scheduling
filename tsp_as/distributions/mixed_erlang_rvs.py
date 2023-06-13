import numpy as np
from numpy.random import Generator
from scipy.stats import erlang


def mixed_erlang_rvs(
    phases: list[int],
    means: list[float],
    weights: list[float],
    num_samples: int,
    rng: Generator,
) -> list[float]:
    """
    Generates samples from a mixed Erlang distribution with two phases.

    Parameters
    ----------
    phases : list[int]
        List of phase parameters for each Erlang distribution.
        NOTE this is the same as the shape parameter, i.e., K.
    means : list[float]
        List of loc parameters for each Erlang distribution.
    weights : list[float]
        List of weights (probabilities) for each Erlang distribution.
    num_samples : int
        Number of samples to generate.
    rng : Generator
        NumPy random number generator.

    Returns
    -------
    np.ndarray[float]
        Array of samples from the mixed Erlang distribution.
    """
    msg = "Input lists must have the same length."
    assert len(means) == len(weights), msg

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Select component Erlang distributions based on weights
    choices = rng.choice(len(weights), p=weights, size=num_samples)

    # Generate samples from the selected Erlang distributions
    samples = [
        erlang.rvs(phases[idx], loc=means[idx], scale=0, random_state=rng)
        for idx in choices
    ]

    return samples
