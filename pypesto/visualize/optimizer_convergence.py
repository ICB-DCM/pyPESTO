import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def optimizer_convergence(result,
                          ax=None,
                          size=(18.5, 10.5)):
    """
    Plot history of optimizer. Can plot either the history of the cost
    function or of the gradient norm, over either the optimizer steps or
    the computation time.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """
    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    fvals = result.optimize_result.get_for_key('fval')
    # ideally use problem.get_reduced_vector here, but checking for nans
    # should also do the trick.
    grads = [
        np.linalg.norm(np.asarray([g for g in grad if not np.isnan(g)]), 2)
        if grad is not None else np.NaN
        for grad in result.optimize_result.get_for_key('grad')
    ]
    msgs = result.optimize_result.get_for_key('message')
    conv_data = pd.DataFrame({
        'fval': fvals,
        'grad': grads,
        'exit message': msgs
    })
    sns.scatterplot(x='fval', y='grad', hue='exit message',
                    data=conv_data, ax=ax)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return ax
