import numpy as np
from scipy.stats import norm


def spectrum(x: np.array,
             nfft: int = None,
             nw: int = None):
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
        nw = int(nfft / 4)

    n_overlap = int(nw / 2)

    # Hanning window
    w = .5 * (1 - np.cos(
        2 * np.pi * np.transpose(np.arange(1, nw+1)) /
        (nw + 1)))
    n = len(x)
    if n < nw:
        x[nw] = 0
        n = nw

    # Number of windows
    k = int((n - n_overlap) / (nw - n_overlap))
    index = np.arange(nw)
    # Normalizing scale factor
    kmu = k * np.linalg.norm(w) ** 2
    spectral_density = np.zeros((nfft))

    for i in range(k):
        xw = np.multiply(w, x[index])
        index += (nw - n_overlap)
        Xx = np.absolute(np.fft.fft(xw, n=nfft, axis=0)) ** 2
        spectral_density += Xx

    # Normalize
    spectral_density = spectral_density * (1 / kmu)

    n2 = int(nfft / 2)
    spectral_density = spectral_density[0:n2]

    return spectral_density


def spectrum0(x: np.array):
    """
    Calculates the spectral density at frequency zero.

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
        spectral_density_zero[:, i] = \
            spectrum(x[:, i], n_samples)[0]
    return spectral_density_zero


def calculate_zscore(chain: np.array,
                     a: float = 0.1,
                     b: float = 0.5):
    """
    Performs a Geweke test on a chain using the first
    "a" fraction and the last "b" fraction of it for
    comparison. Test for equality of the means of the
    first a% and last b% of a Markov chain.

    See:
    Stephen P. Brooks and Gareth O. Roberts.
    Assessing convergence of Markov chain Monte Carlo
    algorithms. Statistics and Computing, 8:319--335, 1998.

    Parameters
    ----------
    chain
    a:
        First fraction of the MCMC chain. Default 10%.
    b:
        Second fraction of the MCMC chain. Default 50%.

    Returns
    -------
    z_score:
        Z-score of the Geweke test.
    p:
        Significance level of the Geweke test.

    """

    nsimu, _ = chain.shape

    # Define First fraction
    index_a = int(a * nsimu)
    # Define Second fraction
    index_b = nsimu - int(b * nsimu) + 1

    # Check if appropiate indexes
    if (index_a + index_b - 1) / nsimu > 1:
        raise ValueError('Error with index_a and index_b')

    # Mean of First fraction
    mean_a = np.mean(chain[0:index_a, :], axis=0)
    # Mean of Second fraction
    mean_b = np.mean(chain[index_b:, :], axis=0)

    # Spectral estimates for variance
    spectrum_a = spectrum0(chain[0:index_a, :])
    spectrum_b = spectrum0(chain[index_b:, :])

    # Calculate z-score
    z_score = (mean_a - mean_b) / (np.sqrt(
        spectrum_a / index_a + spectrum_b /
        (nsimu - index_b + 1)
    ))
    # Calculate significance (p value)
    p = 2 * (1 - norm.cdf(np.absolute(z_score)))

    return z_score, p


def burn_in_by_sequential_geweke(chain: np.array,
                                 zscore: float = 2.):
    """
    Calculates the burn-in of MCMC chains.

    Parameters
    ----------
    chain:
        The MCMC chain after removing warm up phase.
    zscore:
        The Geweke test threshold. Default 2.

    Returns
    -------
    burn_in:
        Iteration where the first and the last fraction
        of the chain do not differ significantly
        regarding Geweke test.

    """

    nsimu, npar = chain.shape
    # number of fragments
    n = 20
    # round each element to the nearest integer
    # toward zero
    e = int(5 * nsimu / 5)
    step = int(e / n)
    ii = np.arange(0, e - 1, step)

    z = np.zeros((len(ii), npar))
    for i in range(len(ii)):
        # Calculate z-score
        z[i, :], _ = calculate_zscore(chain[ii[i]:, :])

    # Sort z-score for Bonferroni-Holm inverse
    # to sorting p-values
    max_z = np.max(np.absolute(z), axis=1)
    idxs = max_z.argsort()[::-1]  # sort descend
    alpha2 = zscore + np.zeros((len(idxs)))

    for i in range(len(max_z)):
        alpha2[idxs[i]] = alpha2[idxs[i]] / \
                          (len(ii) - np.where(idxs == i)[0] + 1)

    if np.where(alpha2 > max_z)[0].size != 0:
        burn_in = (np.where(alpha2 > max_z)[0][0]) * step
    else:
        burn_in = nsimu

    return burn_in
