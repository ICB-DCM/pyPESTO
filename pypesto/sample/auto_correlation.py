import numpy as np


def autocorrelation_sokal(chain):
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

    mx, nx = chain.shape
    tau_est = np.zeros((nx))
    m = np.zeros((nx))

    x = np.fft.fft(chain, axis=0)
    xr = np.real(x)
    xi = np.imag(x)
    xr = xr**2 + xi**2
    xr[0, :] = 0.
    xr = np.real(np.fft.fft(xr, axis=0))
    var = xr[0]/mx/(mx-1)

    for j in range(nx):
        if var[j] == 0.:
            continue
        xr[:, j] = xr[:, j]/xr[0, j]
        sum = -1/3
        for i in range(mx):
            sum = sum + xr[i, j] - 1/6
            if sum < 0.:
                tau_est[j] = 2 * (sum + i/6)
                m[j] = i
                break

    return tau_est
