import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def x_as_ndarray(result):
    """Creates numpy array of size (n_chain, n_sample, n_par).
    Returns ndarray of parameters and fval
    """
    n_chain = len(result['chains'])
    n_sample = len(result['chains'][0])
    n_par = len(result['chains'][0][0]['x'])

    # reserve space
    arr = np.zeros((n_chain, n_sample, n_par))
    arr_fval = np.zeros((n_sample, 1))

    # fill
    for i_chain, chain in enumerate(result['chains']):
        for i_sample, sample in enumerate(chain):
            arr[i_chain, i_sample, :] = sample['x']

    for i_sample in range(0, n_sample):
        arr_fval[i_sample] = result['chains'][0][i_sample]['fval']

    return arr, arr_fval


def sampling_fval(result, size=None, fs = 12, noex_param_name = True):
    """
    Plot parameter values.

    Parameters
    ----------

    results: pypesto.Result or list
        Optimization result obtained by 'optimize.py' or list of those

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    fs: int
        fontsize for the plots

    noex_param_name: bool
        if parameter names exist set: noex_param_name = False

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """
#
    Version2 = True     # results format from Dilan

    # get parameters and fval results as numpy-arrays
    if Version2:
        arr = np.transpose(result['theta'])
        arr_fval = result['log_posterior']
    else:
        arr, arr_fval = x_as_ndarray(result)
        arr = arr[0]

    # plot fval
    ax = plt.subplots(figsize =size)[1]
    x_fval = range(1,len(arr_fval)+1)
    plt.scatter(x_fval, arr_fval, alpha=0.5, s=10)
    ax.set_xlabel('number of interations', fontsize = fs)
    ax.set_ylabel('log Posterior', fontsize = fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.show()

    # plot parameters
    sns.set(style="ticks")
    # transform nparray to pandas for the use of seaborn
    if noex_param_name:
        param_names = []
        for i_param in range(0,np.shape(arr)[1]):
            param_names = np.append(param_names,'par' + str(i_param))

    pd_array = pd.DataFrame(arr, columns = param_names)
    sns.pairplot(pd_array)

    return ax

# sampling_fval(result, size=None, fs = 12, noex_param_name = True)