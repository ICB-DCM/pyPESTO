import logging
import numpy as np


def autocorrelation_sokal(chain):
    """Estimate the integrated autocorrelation time of a time series.
    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
    determine a reasonable window size.
    Args:
        x: The time series. If multidimensional, set the time axis using the
            ``axis`` keyword argument and the function will be computed for
            every other axis.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when the
            chain is too short. If ``True``, give a warning instead of raising
            an :class:`AutocorrError`. (default: ``False``)
    Returns:
        float or array: An estimate of the integrated autocorrelation time of
            the time series ``x`` computed along the axis ``axis``.
    Raises
        AutocorrError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.
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
