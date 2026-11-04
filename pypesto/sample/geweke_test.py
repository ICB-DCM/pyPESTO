"""Helper function for `geweke_test`."""

import logging
import math
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Number of candidate burn-in positions in the sequential Geweke test.
N_FRAGMENTS = 20


def spectrum(x: np.ndarray, nfft: int = None, nw: int = None) -> np.ndarray:
    """
    Power spectral density using Welch's method with a Hann window.

    Deprecated and will be removed in the next meso release. Use
    :func:`spectrum0` for the spectral density at frequency zero.

    Parameters
    ----------
    x:
        Fraction/fragment of one single parameter
        of the MCMC chain.
    nfft:
        The n-point discrete Fourier Transform.
    nw:
        Window length.

    Returns
    -------
    spectral_density:
        The spectral density at the first ``nfft // 2`` Fourier
        frequencies. The input is not centered.
    """
    warnings.warn(
        "`spectrum` is deprecated and will be removed in the next meso "
        "release. Use `spectrum0` for the spectral density at frequency "
        "zero.",
        DeprecationWarning,
        stacklevel=2,
    )
    x = np.asarray(x, dtype=float)
    if nfft is None:
        nfft = min(len(x), 256)
    if nw is None:
        nw = nfft // 4
    step = nw - nw // 2
    n_windows = max((len(x) - nw // 2) // step, 0) if step else 0
    if n_windows == 0:
        return np.full(nfft // 2, np.nan)

    w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, nw + 1) / (nw + 1)))
    windows = sliding_window_view(x, nw)[: (n_windows - 1) * step + 1 : step]
    periodograms = np.abs(np.fft.fft(windows * w, n=nfft, axis=1)) ** 2
    spectral_density = np.sum(periodograms, axis=0) / (n_windows * (w @ w))
    return spectral_density[: nfft // 2]


def spectrum0(x: np.ndarray, nw: int = None) -> np.ndarray:
    """
    Calculate the spectral density at frequency zero.

    This estimates the long-run variance
    S(0) = gamma(0) + 2 * sum_k gamma(k) of each column, so that
    S(0) / n estimates the variance of the column mean. Each column is
    centered with its own mean first. Without centering, the estimate
    picks up a bias proportional to the squared mean.

    The estimate is Welch's method at frequency zero, where the periodogram
    of a tapered window is the squared weighted sum of the window. This is
    batch means with Hann-tapered batches of length `nw` that overlap by
    half.

    Parameters
    ----------
    x:
        Fraction/fragment of the MCMC chain, shape (n_samples, n_par).
    nw:
        Batch length. Defaults to ``floor(sqrt(n_samples))``, so that both
        the batch length and the number of batches grow with the chain
        length, which consistency requires.

    Returns
    -------
    spectral_density_zero:
        Spectral density at zero, shape (n_par,). NaN if no batch of length
        >= 2 fits.
    """
    x = np.asarray(x, dtype=float)
    n_samples, n_par = x.shape
    if nw is None:
        nw = math.isqrt(n_samples)
    step = nw - nw // 2
    n_batches = (n_samples - nw // 2) // step if nw >= 2 else 0
    if n_batches <= 0:
        return np.full(n_par, np.nan)

    # MATLAB's symmetric ``hanning`` window
    w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, nw + 1) / (nw + 1)))
    # (n_batches, n_par, nw): the batches of each column
    batches = sliding_window_view(x - x.mean(axis=0), nw, axis=0)[
        : (n_batches - 1) * step + 1 : step
    ]
    return np.mean((batches @ w) ** 2, axis=0) / (w @ w)


def calculate_zscore(
    chain: np.ndarray, a: float = 0.1, b: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform a Geweke test on a chain.

    Test for equality of the means of the first `a` fraction and the last
    `b` fraction of a Markov chain, with the variance of each mean
    estimated from the spectral density at zero (see :func:`spectrum0`).

    See:
    John Geweke. Evaluating the accuracy of sampling-based approaches to the
    calculation of posterior moments. Bayesian Statistics 4:169--193, 1992.

    Stephen P. Brooks and Gareth O. Roberts.
    Assessing convergence of Markov chain Monte Carlo
    algorithms. Statistics and Computing, 8:319--335, 1998.

    Parameters
    ----------
    chain:
        The MCMC chain, shape (n_samples, n_par).
    a:
        First fraction of the MCMC chain.
    b:
        Second fraction of the MCMC chain.

    Returns
    -------
    z_score:
        Z-score of the Geweke test per parameter, shape (n_par,). 0 if both means are equal, also when
        their variance estimate is 0; +-inf if only the variance is 0;
        NaN if the fractions are too short to estimate the variance.
    p:
        P-value of the Geweke test per parameter, shape (n_par,).
    """
    if not (0 < a < 1 and 0 < b < 1):
        raise ValueError("The fractions a and b must lie in (0, 1).")
    if a + b > 1:
        raise ValueError(
            "The first and last fraction of the chain overlap (a + b > 1)."
        )

    nsamples, _ = chain.shape
    n_a = int(np.floor(a * nsamples))
    n_b = int(np.floor(b * nsamples))
    first = chain[:n_a, :]
    last = chain[nsamples - n_b :, :]

    # Expect to see RuntimeWarnings in this block for short or constant
    # chains
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_diff = np.mean(first, axis=0) - np.mean(last, axis=0)
        variance = spectrum0(first) / n_a + spectrum0(last) / n_b
        z_score = np.where(mean_diff == 0, 0.0, mean_diff / np.sqrt(variance))
        p = 2 * norm.sf(np.absolute(z_score))

    return z_score, p


def _combine_parameters(p: np.ndarray) -> np.ndarray:
    """
    Combine the parameters' p-values of each fragment (Bonferroni).

    A fragment's null hypothesis is "this sub-chain is stationary in all
    parameters". ``min(1, npar * min_j p_j)`` is a valid p-value for it for
    any dependence between the parameters, unlike the single-test p-value of
    the largest |z|, which rejects too often for many parameters.

    Parameters
    ----------
    p:
        Geweke p-values, shape (n_fragments, npar). NaN (too short to test)
        counts as significant.

    Returns
    -------
    One p-value per fragment.
    """
    p = np.nan_to_num(p, nan=0.0)
    return np.minimum(1.0, p.shape[1] * np.min(p, axis=1))


def _first_not_rejected(p_values: np.ndarray, zscore: float) -> int | None:
    """
    Holm's step-down procedure over the fragments of the sequential test.

    Each fragment's null hypothesis is "this sub-chain is stationary", with
    p-value from :func:`_combine_parameters`. The family-wise level is
    ``alpha = 2 * (1 - Phi(zscore))``.

    See:
    Sture Holm. A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics, 6(2):65--70, 1979.

    Parameters
    ----------
    p_values:
        P-value per fragment, without NaN.
    zscore:
        Single-test z threshold defining the family-wise level.

    Returns
    -------
    Index of the first fragment (in chain order) that is not rejected, or
    None if all are rejected.
    """
    n_tests = len(p_values)
    alpha = 2 * norm.sf(zscore)

    rejected = np.zeros(n_tests, dtype=bool)
    for rank, idx in enumerate(np.argsort(p_values, kind="stable")):
        if p_values[idx] > alpha / (n_tests - rank):
            break
        rejected[idx] = True

    not_rejected = np.flatnonzero(~rejected)
    return int(not_rejected[0]) if len(not_rejected) else None


def burn_in_by_sequential_geweke(
    chain: np.ndarray, zscore: float = 2.0
) -> int:
    """
    Calculate the burn-in of MCMC chains.

    The Geweke test is applied to the chain without its first
    ``floor(i * n / 20)`` samples, i = 0, ..., 19, and the burn-in is the first of these starting
    points that is not rejected, with Holm's correction for testing all
    starting points. Each parameter is tested separately, and their
    p-values are combined per starting point with Bonferroni's correction.

    See:
    Benjamin Ballnus et al. Comprehensive benchmarking of Markov chain
    Monte Carlo methods for dynamical systems. BMC Systems Biology 11:63,
    2017 (Additional file 1, Section 2).

    Parameters
    ----------
    chain:
        The MCMC chain after removing warm up phase.
    zscore:
        The Geweke test threshold for a single test. It sets the
        family-wise significance level ``2 * (1 - Phi(zscore))``.

    Returns
    -------
    burn_in:
        Iteration where the first and the last fraction
        of the chain do not differ significantly
        regarding Geweke test.
    """
    nsamples = chain.shape[0]
    if nsamples < N_FRAGMENTS:
        return _not_converged(nsamples)

    starts = np.arange(N_FRAGMENTS) * nsamples // N_FRAGMENTS
    p = np.array([calculate_zscore(chain[start:, :])[1] for start in starts])
    first = _first_not_rejected(_combine_parameters(p), zscore)
    if first is None:
        return _not_converged(nsamples)
    return int(starts[first])


def _not_converged(nsamples: int) -> int:
    """Warn that the chain has not converged; its length is the burn-in."""
    logger.warning(
        "Burn in index coincides with chain "
        "length. The chain seems to not have "
        "converged yet.\n"
        "You may want to use a larger number "
        "of samples."
    )
    return nsamples
