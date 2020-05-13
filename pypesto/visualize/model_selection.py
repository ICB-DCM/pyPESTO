import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import pandas as pd
from typing import Tuple


def plot_modelselection(
        selection_history : dict,
        aic_or_bic : str = 'AIC',
        mode : str = 'delta',
        fz : int = 14,
        size : Tuple[float, float] = [5,4]):
    """
    Plot AIC or BIC for different models selected during model selection
    routine.

    Parameters
    ----------
    selection_history: dict
        pyPESTO result of model selection pipeline.
    aic_or_bic: 'AIC'|'BIC'
        plot AIC or BIC
    mode: 'delta'|'absolute'
        plot delta AIC (BIC) => e.g., AIC - min(AIC), or absolute AIC (BIC)
        values
    fz: int
        fontsize
    size: ndarray
        Figure size in inches.

    Returns
    -------
    ax:
        The plot axes.
    """
    # create pandas dataframe from dictionary selection_history
    df_cols = ['AIC', 'BIC', 'xlabel', 'chosen_model', 'delta_AIC',
               'delta_BIC']
    df = pd.DataFrame([],
                      columns=df_cols,
                      index=range(len(selection_history)), dtype=float)
    for ind, i_k in enumerate(selection_history):
        df.iloc[ind, df_cols.index('AIC')] = selection_history[i_k]['AIC']
        df.iloc[ind, df_cols.index('BIC')] = selection_history[i_k]['BIC']
        df.iloc[ind, df_cols.index('xlabel')] = \
            selection_history[i_k]['compared_model_id']
        df.iloc[ind, df_cols.index('chosen_model')] = i_k

    df['delta_AIC'] = df['AIC'] - df['AIC'].min()
    df['delta_BIC'] = df['BIC'] - df['BIC'].min()

    # how many models shall be plotted on the x-axis
    nr_models = int((len(selection_history)+1)/2)
    # dataframe of values which shall be plotted
    df_to_plot = pd.DataFrame([],
                              columns=df_cols,
                              index=range(nr_models),dtype=float)

    which_model = 'PYPESTO_INITIAL_MODEL'
    for i_m in range(nr_models):
        # get entries for model of interest (=which_model)
        df1 = df[df['xlabel'] == which_model]
        # get index of minimum AIC for the model of interest (=which_model)
        idx_min_AIC1 = df1['AIC'].idxmin()
        # copy row of interest into dataframe which shall be plotted
        df_to_plot.iloc[i_m,:] = df.iloc[idx_min_AIC1,:]

        # rename 'PYPESTO_INITIAL_MODEL'
        if which_model == 'PYPESTO_INITIAL_MODEL':
            df_to_plot['xlabel'] = 'Initial model search'
        # get next model
        which_model = df.iloc[idx_min_AIC1, df_cols.index('chosen_model')]

    # define what to plot and set y-label
    if aic_or_bic == 'AIC':
        if mode == 'delta':
            col_to_plot = 'delta_AIC'
            mode = '$\Delta$'
        elif mode == 'absolute':
            col_to_plot = 'AIC'
    elif aic_or_bic == 'BIC':
        if mode == 'delta':
            col_to_plot = 'delta_BIC'
            mode = '$\Delta$'
        elif mode == 'absolute':
            col_to_plot = 'BIC'

    # FIGURE
    fig, ax = plt.subplots(figsize=size)
    width = 0.75
    ax.bar(df_to_plot.index.values, df_to_plot[col_to_plot].values,
           width, color='lightgrey', edgecolor='k')

    ax.get_xticks()
    ax.set_xticks(df_to_plot.index.values)
    ax.set_ylabel(mode + ' ' + aic_or_bic, fontsize = fz)
    ax.set_xticklabels(df_to_plot.xlabel.values, fontsize = fz-2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fz-2)
    ytl = ax.get_yticks()
    ax.set_ylim([0,max(ytl)])
    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax
