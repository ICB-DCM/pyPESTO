"""Helper function for `geweke_test`."""

import logging
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Number of candidate burn-in positions in the sequential Geweke test.
N_FRAGMENTS = 20


def _hann_window(nw: int) -> np.ndarray:
    """Symmetric Hann window of length `nw` (MATLAB's ``hanning``)."""
    t = np.arange(1, nw + 1)
    return 0.5 * (1 - np.cos(2 * np.pi * t / (nw + 1)))


def _welch_layout(n: int, nw: int) -> tuple[int, int]:
    """Step between window starts and number of windows (50% overlap)."""
    step = nw - nw // 2
    if step == 0:
        return 0, 0
    return step, max((n - nw // 2) // step, 0)


def _spectral_density_at_zero(x: np.ndarray, nw: int) -> np.ndarray:
    """
    Welch estimate of the spectral density at frequency zero.

    At frequency zero, the periodogram of a tapered window is the squared
    weighted sum of the window. This is batch means with overlapping,
    Hann-tapered batches of length `nw`. The input is used as is (no
    centering).

    Parameters
    ----------
    x:
        Array of shape (n_samples, n_par).
    nw:
        Window (batch) length.

    Returns
    -------
    Array of shape (n_par,). NaN if no window of length >= 2 fits.
    """
    n, n_par = x.shape
    step, k = _welch_layout(n, nw)
    if nw < 2 or k == 0:
        return np.full(n_par, np.nan)
    w = _hann_window(nw)
    # (k, n_par, nw): the k windows of each column
    windows = sliding_window_view(x, nw, axis=0)[: (k - 1) * step + 1 : step]
    weighted_sums = windows @ w
    return np.mean(weighted_sums**2, axis=0) / (w @ w)


def spectrum(x: np.ndarray, nfft: int = None, nw: int = None) -> np.ndarray:
    """
    Power spectral density using Welch's method with a Hann window.

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
    x = np.asarray(x, dtype=float)
    if nfft is None:
        nfft = min(len(x), 256)
    if nw is None:
        nw = nfft // 4

    step, k = _welch_layout(len(x), nw)
    if k == 0:
        return np.full(nfft // 2, np.nan)

    w = _hann_window(nw)
    windows = sliding_window_view(x, nw)[: (k - 1) * step + 1 : step] * w
    periodograms = np.abs(np.fft.fft(windows, n=nfft, axis=1)) ** 2
    spectral_density = np.sum(periodograms, axis=0) / (k * (w @ w))
    return spectral_density[: nfft // 2]


def spectrum0(x: np.ndarray, nw: int = None) -> np.ndarray:
    """
    Calculate the spectral density at frequency zero.

    This estimates the long-run variance
    S(0) = gamma(0) + 2 * sum_k gamma(k) of each column, so that
    S(0) / n estimates the variance of the column mean. Each column is
    centered with its own mean first. Without centering, the estimate
    picks up a bias proportional to the squared mean.

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
        Spectral density at zero, shape (1, n_par).
    """
    x = np.asarray(x, dtype=float)
    n_samples = x.shape[0]
    if nw is None:
        nw = int(np.floor(np.sqrt(n_samples)))
    centered = x - x.mean(axis=0) if n_samples else x
    return _spectral_density_at_zero(centered, nw)[np.newaxis, :]


def calculate_zscore(
    chain: np.ndarray, a: float = 0.1, b: float = 0.5
) -> tuple[float, float]:
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
    chain
    a:
        First fraction of the MCMC chain.
    b:
        Second fraction of the MCMC chain.

    Returns
    -------
    z_score:
        Z-score of the Geweke test. 0 if both means are equal, also when
        their variance estimate is 0; +-inf if only the variance is 0;
        NaN if the fractions are too short to estimate the variance.
    p:
        Significance level of the Geweke test.
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
    with (
        warnings.catch_warnings(),
        np.errstate(divide="ignore", invalid="ignore"),
    ):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_diff = np.mean(first, axis=0) - np.mean(last, axis=0)
        variance = spectrum0(first) / n_a + spectrum0(last) / n_b
        z_score = np.where(mean_diff == 0, 0.0, mean_diff / np.sqrt(variance))
        p = 2 * norm.sf(np.absolute(z_score))

    return z_score, p


def _first_not_rejected(max_abs_z: np.ndarray, zscore: float) -> int | None:
    """
    Holm's step-down procedure over the fragments of the sequential test.

    Each fragment's null hypothesis is "this sub-chain is stationary", with
    p-value from the largest |z| over all parameters. The family-wise level
    is ``alpha = 2 * (1 - Phi(zscore))``.

    See:
    Sture Holm. A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics, 6(2):65--70, 1979.

    Parameters
    ----------
    max_abs_z:
        Largest absolute z-score per fragment. NaN counts as rejected.
    zscore:
        Single-test z threshold defining the family-wise level.

    Returns
    -------
    Index of the first fragment (in chain order) that is not rejected, or
    None if all are rejected.
    """
    n_tests = len(max_abs_z)
    alpha = 2 * norm.sf(zscore)
    p_values = 2 * norm.sf(np.nan_to_num(max_abs_z, nan=np.inf))

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

    The Geweke test is applied to the chain without its first 0, 1/20,
    2/20, ... of samples, and the burn-in is the first of these starting
    points that is not rejected, with Holm's correction for testing all
    starting points. Each parameter is tested separately and the largest
    |z| counts.

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
    nsamples, npar = chain.shape

    first = None
    if nsamples >= N_FRAGMENTS:
        step = nsamples // N_FRAGMENTS
        fragments = np.arange(0, nsamples - 1, step)

        z = np.zeros((len(fragments), npar))
        for i, start in enumerate(fragments):
            z[i, :], _ = calculate_zscore(chain[start:, :])

        first = _first_not_rejected(np.max(np.absolute(z), axis=1), zscore)

    if first is None:
        logger.warning(
            "Burn in index coincides with chain "
            "length. The chain seems to not have "
            "converged yet.\n"
            "You may want to use a larger number "
            "of samples."
        )
        return nsamples

    return int(fragments[first])
