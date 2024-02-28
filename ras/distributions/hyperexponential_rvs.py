import numpy as np
from numpy.random import Generator


def hyperexponential_rvs(
    scales: list[float], weights: list[float], num_samples: int, rng: Generator
) -> list[float]:
    """
    Generates samples from a hyperexponential distribution consisting of two
    exponential distributions.

    Parameters
    ----------
    scales
        List of scale (mean) parameters for each exponential distribution.
    weights
        List of weights (probabilities) for each exponential distribution.
    num_samples
        Number of samples to generate.
    rng
        NumPy random number generator.

    Returns
    -------
    np.ndarray
        Array of samples from the hyperexponential distribution.
    """
    # Normalize weights to probabilities
    probs = np.array(weights) / sum(weights)

    # Select component exponential distributions based on weights
    choices = rng.choice(len(probs), p=probs, size=num_samples)

    # Generate samples from the selected exponential distributions
    samples = []
    for idx in np.unique(choices):
        num_choice = np.sum(choices == idx)
        samples.extend(rng.exponential(scales[idx], size=num_choice))

    return samples
