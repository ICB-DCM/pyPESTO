"""Helper function for `geweke_test`."""

import logging
import warnings

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def spectrum(x: np.ndarray, nfft: int = None, nw: int = None) -> np.ndarray:
    """
    Power spectral density using Hanning window.

    Parameters
    ----------
    x:
        Fraction/fragment of one single parameter
        of the MCMC chain.
    nfft:
        The n-point discrete Fourier Transform.
    nw:
        Number of windows.

    Returns
    -------
    spectral_density:
        The spectral density.
    """
    if nfft is None:
        nfft = np.min(len(x), 256)

    if nw is None:
        nw = np.floor(nfft / 4).astype(int)

    n_overlap = np.floor(nw / 2).astype(int)

    # Hanning window
    w = 0.5 * (
        1 - np.cos(2 * np.pi * np.transpose(np.arange(1, nw + 1)) / (nw + 1))
    )
    n = len(x)
    if n < nw:
        x[nw] = 0
        n = nw

    # Number of windows
    k = (
        np.floor((n - n_overlap) / (nw - n_overlap)).astype(int)
        if nw != n_overlap
        else 0
    )
    index = np.arange(nw)
    # Normalizing scale factor
    kmu = k * np.linalg.norm(w) ** 2
    spectral_density = np.zeros(nfft)

    for _ in range(k):
        xw = w * x[index]
        index += nw - n_overlap
        Xx = np.absolute(np.fft.fft(xw, n=nfft, axis=0)) ** 2
        spectral_density += Xx

    n2 = np.floor(nfft / 2).astype(int)

    spectral_density = spectral_density[0:n2]

    # Normalize
    if kmu != 0:
        spectral_density = spectral_density * (1 / kmu)
    else:
        spectral_density = np.full(spectral_density.shape, np.nan)

    return spectral_density


def spectrum0(x: np.ndarray) -> np.ndarray:
    """
    Calculate the spectral density at frequency zero.

    Parameters
    ----------
    x:
        Fraction/fragment of the MCMC chain.

    Returns
    -------
    spectral_density_zero:
        Spectral density at zero.
    """
    n_samples, n_par = x.shape
    spectral_density_zero = np.zeros((1, n_par))

    for i in range(n_par):
        _spectral_density_zero = spectrum(x[:, i], n_samples)
        if len(_spectral_density_zero) > 0:
            spectral_density_zero[:, i] = _spectral_density_zero[0]
    return spectral_density_zero


def calculate_zscore(
    chain: np.ndarray, a: float = 0.1, b: float = 0.5
) -> tuple[float, float]:
    """
    Perform a Geweke test on a chain.

    Use the first "a" fraction and the last "b" fraction of it for
    comparison. Test for equality of the means of the first a% and last b%
    of a Markov chain.

    See:
    Stephen P. Brooks and Gareth O. Roberts.
    Assessing convergence of Markov chain Monte Carlo
    algorithms. Statistics and Computing, 8:319--335, 1998.

    Parameters
    ----------
    chain
    a:
        First fraction of the MCMC chain.
    b:
        Second fraction of the MCMC chain.

    Returns
    -------
    z_score:
        Z-score of the Geweke test.
    p:
        Significance level of the Geweke test.
    """
    nsamples, _ = chain.shape

    # Define First fraction
    index_a = np.floor(a * nsamples).astype(int)
    # Define Second fraction
    index_b = nsamples - np.floor(b * nsamples).astype(int) + 1

    # Check if appropiate indexes
    if (index_a + index_b) / nsamples > 1:
        raise ValueError(
            "Sample size too small to "
            "meaningfully extract subsets "
            "for Geweke's test."
        )

    # Expect to see RuntimeWarnings in this block for short chains
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Mean of First fraction
        mean_a = np.mean(chain[0:index_a, :], axis=0)
        # Mean of Second fraction
        mean_b = np.mean(chain[index_b:, :], axis=0)

        # Spectral estimates for variance
        spectrum_a = spectrum0(chain[0:index_a, :])
        spectrum_b = spectrum0(chain[index_b:, :])

        # Calculate z-score
        z_score = (mean_a - mean_b) / (
            np.sqrt(
                spectrum_a / index_a + spectrum_b / (nsamples - index_b + 1)
            )
        )
        # Calculate significance (p value)
        p = 2 * (1 - norm.cdf(np.absolute(z_score)))

    return z_score, p


def burn_in_by_sequential_geweke(
    chain: np.ndarray, zscore: float = 2.0
) -> int:
    """
    Calculate the burn-in of MCMC chains.

    Parameters
    ----------
    chain:
        The MCMC chain after removing warm up phase.
    zscore:
        The Geweke test threshold.

    Returns
    -------
    burn_in:
        Iteration where the first and the last fraction
        of the chain do not differ significantly
        regarding Geweke test.
    """
    nsamples, npar = chain.shape
    # number of fragments
    n = 20
    # round each element to the nearest integer
    # toward zero
    step = np.floor(nsamples / n).astype(int)
    fragments = np.arange(0, nsamples - 1, step)

    z = np.zeros((len(fragments), npar))
    for i, indices in enumerate(fragments):
        # Calculate z-score
        z[i, :], _ = calculate_zscore(chain[indices:, :])

    # Sort z-score for Bonferroni-Holm inverse
    # to sorting p-values
    max_z = np.max(np.absolute(z), axis=1)
    idxs = max_z.argsort()[::-1]  # sort descend
    alpha2 = zscore * np.ones(len(idxs))

    for i in range(len(max_z)):
        alpha2[idxs[i]] /= len(fragments) - np.argwhere(idxs == i).item(0) + 1

    if np.any(alpha2 > max_z):
        burn_in = (np.where(alpha2 > max_z)[0][0]) * step
    else:
        burn_in = nsamples
        logger.warning(
            "Burn in index coincides with chain "
            "length. The chain seems to not have "
            "converged yet.\n"
            "You may want to use a larger number "
            "of samples."
        )

    return burn_in
