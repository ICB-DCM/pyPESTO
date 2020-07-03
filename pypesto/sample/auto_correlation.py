import numpy as np


def autocorrelation_sokal(chain: np.array):
    """
    Estimate the integrated autocorrelation time of a MCMC chain
    using Sokal's adaptive truncated periodogram estimator.

    Parameters
    ----------
        chain: The MCMC chain.
    Returns
    -------
        tau_est: An estimate of the integrated autocorrelation time of
        the MCMC chain.
    """

    nsamples, npar = chain.shape
    tau_est = np.zeros((npar))

    # Calculate the fast Fourier transform
    x = np.fft.fft(chain, axis=0)
    # Get the real part
    xr = np.real(x)
    # Get the imaginary part
    xi = np.imag(x)

    xr = xr**2 + xi**2
    # First value to zero
    xr[0, :] = 0.
    # Calculate the fast Fourier transform
    # of the transformation
    xr = np.real(np.fft.fft(xr, axis=0))
    # Calculate the variance
    var = xr[0]/nsamples/(nsamples-1)

    # Loop over parameters
    for j in range(npar):
        if var[j] == 0.:
            continue
        # Normalize by the first value
        xr[:, j] = xr[:, j]/xr[0, j]
        # Initiate variable
        _sum = -1/3
        # Loop over samples
        for i in range(nsamples):
            _sum = _sum + xr[i, j] - 1/6
            if _sum < 0.:
                tau_est[j] = 2 * (_sum + i/6)
                break

    return tau_est
