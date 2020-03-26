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

# ####################################################################


def get_data_to_plot(result, options, noex_param_name=True):
    Version2 = True     # results format from Dilan TODO: TO DISCUSS

    # get parameters and fval results as numpy-arrays
    if Version2:
        arr_param = np.transpose(result['theta'])
        arr_fval = result['log_posterior']
        theta_lb = options['theta_bounds_lower']
        theta_ub = options['theta_bounds_upper']

    else:
        arr_param, arr_fval = x_as_ndarray(result)
        arr_param = arr_param[0]    # only plot the first chain

    # get parameter names
    if noex_param_name: # for now: it does not exist, so create param names
        param_names = []
        for i_param in range(0,np.shape(arr_param)[1]):
            param_names = np.append(param_names,'par' + str(i_param))

    # transform nparray to pandas for the use of seaborn
    pd_params = pd.DataFrame(arr_param, columns=param_names)
    pd_fval = pd.DataFrame(data=np.transpose(arr_fval),
                           columns=['logPosterior'])
    pd_iter = pd.DataFrame(data=np.transpose(range(1, len(arr_fval) + 1)),
                           columns=['iteration'])
    params_fval = pd.concat([pd_params,pd_fval,pd_iter],
                            axis=1, ignore_index=False)

#
    # some global parameters
    # nr_iter = range(1, len(arr_fval) + 1)   # number of iterations
    nr_params = arr_param.shape[1]                # number of parameters

    return nr_params, params_fval, theta_lb, theta_ub


def sampling_fval(result, options, size=None, fs = 12):
    """
    Plot logPosterior over iterations.

    Parameters
    ----------
    """
    # get data which should be plotted
    _, params_fval, _, _= get_data_to_plot(result,
                                                    options,
                                                    noex_param_name=True)

    fig, ax = plt.subplots(figsize =size)

    sns.set(style="ticks")
    kwargs = {'edgecolor': "w",  # for edge color
              'linewidth': 0.3,
              's': 10}
    ax = sns.scatterplot(x="iteration", y="logPosterior", data=params_fval,
                         **kwargs)

    # ax.scatter(nr_iter, arr_fval, alpha=0.5, s=10)
    ax.set_xlabel('number of iterations', fontsize = fs)
    ax.set_ylabel('log Posterior', fontsize = fs)
    # ax.tick_params(axis='both', which='major', labelsize=fs)

    sns.despine()

    plt.show()

    return ax


def sampling_parameters(result, options, size=None, fs = 12):
    '''
    Plot parameter over iterations
    :param result:
    :param options:
    :param size:
    :param fs:
    :param noex_param_name:
    :return:
    '''
    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub = \
        get_data_to_plot(result,options,noex_param_name=True)
    param_names = params_fval.columns.values[0:nr_params]

    # compute, how many rows and columns we need for the subplots
    num_row = int(np.round(np.sqrt(nr_params)))
    num_col = int(np.ceil(nr_params / num_row))

    fig, axes = plt.subplots(num_row,num_col, squeeze=False, figsize=size)
    axes = dict(zip(param_names, axes.flat))

    sns.set(style="ticks")
    kwargs = {'edgecolor': "w",  # for edge color
              'linewidth': 0.3,
              's': 10}
    # fig, ax = plt.subplots(nr_params, figsize=size)[1]
    for idx, plot_id in enumerate(param_names):
        ax = axes[plot_id]
        ax = sns.scatterplot(x="iteration", y=plot_id, data=params_fval, ax=ax,
                             **kwargs)
        # ax.scatter(nr_iter, params_fval[plot_id], alpha=0.5, s=10)
        ax.set_xlabel('number of iterations', fontsize=fs)
        ax.set_ylabel(param_names[idx], fontsize=fs)
        ax.set_ylim([theta_lb[idx],theta_ub[idx]])
        # ax.tick_params(axis='both', which='major', labelsize=fs)
    fig.tight_layout()
    sns.despine()
    plt.show()

    return ax


def sampling_parameter_corr(result, options, size=None, fs=12):
    '''
        Plot parameter correlations
        :param result:
        :param options:
        :param size:
        :param fs:
        :return:
        '''
    #
    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub = \
        get_data_to_plot(result, options, noex_param_name=True)

    sns.set(style="ticks")
    # df1 = df1.drop(['B', 'C'], axis=1)
    ax = sns.pairplot(params_fval.drop(['logPosterior', 'iteration'], axis=1))

    return ax


def sampling_marginal(result,options,size,fs):
    '''
    Plot Marginals
    :param result:
    :param options:
    :param size:
    :param fs:
    :return:
    '''
    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub = \
        get_data_to_plot(result, options, noex_param_name=True)
    param_names = params_fval.columns.values[0:nr_params]

    # compute, how many rows and columns we need for the subplots
    num_row = int(np.round(np.sqrt(nr_params)))
    num_col = int(np.ceil(nr_params / num_row))

    fig, axes = plt.subplots(num_row, num_col, squeeze=False, figsize=size)
    axes = dict(zip(param_names, axes.flat))

    # fig, ax = plt.subplots(nr_params, figsize=size)[1]
    for idx, plot_id in enumerate(param_names):

        ax = axes[plot_id]
        # sns.set_style('ticks')
        sns.set(style="ticks")
        ax = sns.kdeplot(params_fval[plot_id], bw=0.5, ax=ax)
        ax.set_xlabel('log(' + param_names[idx] + ')')
        ax.set_ylabel('Density')
        sns.despine()

    fig.tight_layout()
    plt.show()

    return ax

