import numpy as np
from numba import jit


@jit
def sum_fast(nsimu: int,
             x: np.array,
             tau: float = 0.,
             sum: float = -1/3):

    for i in range(nsimu):
        sum = sum + x[i] - 1/6
        if sum < 0:
            tau = 2 * (sum + i/6)
            break

    return tau


def auto_correlation(chain: np.array):
    """
    This function estimates the integrated autocorrelation time
    using Sokal's adaptive truncated periodogram estimator.

    Parameters
    ----------
    chain:
        The MCMC parameter samples without warm up phase.
        In case of sampling with parallel tempering, the
        chain with the lowest temperature T=1 should be passed.

    Returns
    -------
    tau:
        Array with the auto-correlation time tau for each parameter
        dimension. We suggest taking the maximum over all components.

    """

    nsimu, npar = chain.shape
    tau = np.zeros((npar))

    # Fast fourier transform
    x = np.fft.fft(chain, axis=0)
    # Take real part
    x_real = x.real
    # Take imaginary part
    x_imag = x.imag
    x_real = x_real**2+x_imag**2
    x_real[0, :] = 0.
    # Fast fourier transform of the real part
    x_real = np.fft.fft(x_real, axis=0).real
    # Variance
    var = x_real[0, :]/(nsimu)/(nsimu-1)

    # Loop over parameters
    for j in range(npar):
        if var[j] == 0:
            continue
        x_real[:, j] = x_real[:, j]/x_real[0, j]
        tau[j] = sum_fast(nsimu, x_real[:, j])

    return tau
