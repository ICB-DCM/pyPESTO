import numpy as np


def autocorrelation_sokal(chain: np.ndarray) -> np.ndarray:
    """
    Estimate the integrated autocorrelation time of a MCMC chain.

    Uses Sokal's adaptive truncated periodogram estimator.

    - Haario, H., Laine, M., Mira, A. et al. DRAM: Efficient
    adaptive MCMC. Stat Comput 16, 339–354 (2006).
    https://doi.org/10.1007/s11222-006-9438-0

    - Sokal A. (1997) Monte Carlo Methods in Statistical Mechanics:
    Foundations and New Algorithms. In: DeWitt-Morette C.,
    Cartier P., Folacci A. (eds) Functional Integration.
    NATO ASI Series (Series B: Physics), vol 361. Springer, Boston, MA

    Parameters
    ----------
        chain: The MCMC chain.

    Returns
    -------
        tau_est: An estimate of the integrated autocorrelation time of
        the MCMC chain.
    """
    nsamples, npar = chain.shape
    tau_est = np.zeros(npar)

    # Calculate the fast Fourier transform
    x = np.fft.fft(chain, axis=0)
    # Get the real part
    xr = np.real(x)
    # Get the imaginary part
    xi = np.imag(x)

    xr = xr**2 + xi**2
    # First value to zero
    xr[0, :] = 0.0
    # Calculate the fast Fourier transform
    # of the transformation
    xr = np.real(np.fft.fft(xr, axis=0))
    # Calculate the variance
    var = xr[0] / nsamples / (nsamples - 1)

    # Loop over parameters
    for j in range(npar):
        if var[j] == 0.0:
            continue
        # Normalize by the first value
        xr[:, j] = xr[:, j] / xr[0, j]
        # Initiate variable
        _sum = -1 / 3
        # Loop over samples
        for i in range(nsamples):
            _sum = _sum + xr[i, j] - 1 / 6
            if _sum < 0.0:
                tau_est[j] = 2 * (_sum + i / 6)
                break

    return tau_est
