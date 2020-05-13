import numpy as np
from scipy.stats import norm


def spectrum(x, nfft=None, nw=None):
    '''SPECTRUM Power spectral density using Hanning window
    %  [y,f]=spectrum(x,nfft,nw)

    % See also: psd.m in Signal Processing Toolbox

    % Marko Laine <Marko.Laine@Helsinki.FI>
    % $Revision: 1.3 $  $Date: 2007/08/10 08:54:49 $
    '''

    if nfft is None:
        nfft = np.min(len(x), 256)

    if nw is None:
        nw = int(nfft / 4)

    noverlap = int(nw / 2)

    # Hanning window
    w = .5 * (1 - np.cos(2 * np.pi * np.transpose(np.arange(nw)) / (nw + 1)))
    n = len(x)
    if n < nw:
        x[nw] = 0
        n = nw

    k = int((n - noverlap) / (nw - noverlap))  # no of windows
    index = np.arange(nw)
    kmu = k * np.linalg.norm(w) ** 2  # Normalizing scale factor
    y = np.zeros((nfft))
    for i in range(k):
        xw = np.multiply(w, x[index])
        index += (nw - noverlap)
        Xx = np.absolute(np.fft.fft(xw, nfft)) ** 2
        y += Xx

    y = y * (1 / kmu)  # normalize

    n2 = int(nfft / 2)
    y = y[0:n2]

    return y


def spectrum0(x):
    '''SPECTRUM0 Spectral density at frequency zero
    spectrum0(x) spectral density at zero for columns of x
    '''
    m, n = x.shape
    s = np.zeros((1, n))
    #     print(m)
    for i in range(n):
        s[:, i] = spectrum(x[:, i], m)[0]
    return s


def gewekeTest(chain, a=0.1, b=0.5):
    '''GEWEKE Geweke's MCMC convergence diagnostic
     Published & implemented by Marko Laine (2006) - used by Ballnus et al.
     (2017). Performs a Geweke test on a chain using the first a fraction and
     the last b fraction of it for comparison. Returns test value z and p
     value p.
     [z,p] = geweke(chain,a,b)
     Test for equality of the means of the first a% (default 10%) and
     last b% (50%) of a Markov chain.
     See:
     Stephen P. Brooks and Gareth O. Roberts.
     Assessing convergence of Markov chain Monte Carlo algorithms.
     Statistics and Computing, 8:319--335, 1998.
    '''

    nsimu, _ = chain.shape

    # Define First fragment
    na = int(a * nsimu)
    # Define Second fragment
    nb = nsimu - int(b * nsimu) + 1

    # Check if appropiate indexes
    if (na + nb - 1) / nsimu > 1:
        raise ValueError('Error with na and nb')

    # Mean of First fragment
    m1 = np.mean(chain[0:na, :], axis=0)
    # Mean of Second fragment
    m2 = np.mean(chain[nb:, :], axis=0)

    # Spectral estimates for variance
    sa = spectrum0(chain[0:na, :])
    sb = spectrum0(chain[nb:, :])

    z = (m1 - m2) / (np.sqrt(sa / na + sb / (nsimu - nb + 1)))
    p = 2 * (1 - norm.cdf(np.absolute(z)))

    return z, p


def burnInBySequentialGeweke(chain, zscore=2.):
    nsimu, npar = chain.shape
    # number of fragments
    n = 20
    # round each element to the nearest integer toward zero.
    e = int(5 * nsimu / 5)
    l = int(e / n)
    ii = np.arange(0, e - 1, l)

    z = np.zeros((len(ii), npar))
    for i in range(len(ii)):
        z[i, :], _ = gewekeTest(chain[ii[i]:, :])

    # Sort z-score for Bonferroni-Holm inverse to sorting p-values
    max_z = np.max(np.absolute(z), axis=1)

    idxs = max_z.argsort()[::-1]  # sort descend

    alpha2 = zscore + np.zeros((len(idxs)))

    for i in range(len(max_z)):
        alpha2[idxs[i]] = alpha2[idxs[i]] / (len(ii) - np.where(idxs == i)[0] + 1)

    if np.where(alpha2 > max_z)[0].size != 0:
        burn_in = (np.where(alpha2 > max_z)[0][0]) * l
    else:
        burn_in = nsimu

    return burn_in
