import numpy as np


def auto_correlation(chain):
    '''Published & implemented by Marko Laine (2006) - used by Ballnus et al.
    (2016). This function calculates the auto-correlation time tau for one parameter
    dimension. We suggest taking the maximum over all components. The
    effective sample size is determined counting the remaining points after
    thinning the signal by tau.
    This function estimates the integrated autocorrelation time
    using Sokal's adaptive truncated periodogram estimator.'''

    nsimu, npar = chain.shape
    tau = np.zeros((npar))

    x = np.transpose(np.fft.fft(np.transpose(chain))) # to have same behavior as in matlab
    # Take real part
    x_real = x.real
    # Take imaginary part
    x_imag = x.imag
    x_real = x_real**2+x_imag**2
    x_real[0,:] = 0.
    x_real = np.transpose((np.fft.fft(np.transpose(x_real)))).real
    var = x_real[0,:]/(nsimu+1)/(nsimu)

    for j in range(npar):
        if var[j] == 0:
            continue
        x_real[:,j] = x_real[:,j]/x_real[0,j]
        sum = -1/3 # initialize
        for i in range(nsimu):
            sum = sum + x_real[i,j]-1/6
            if sum < 0:
                tau[j] = 2*(sum+i/6)
                break
    return tau