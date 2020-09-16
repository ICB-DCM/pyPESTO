import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pypesto

from typing import Optional, Tuple


def optimizer_convergence(result: pypesto.Result,
                          ax: Optional[plt.Axes] = None,
                          xscale: str = 'symlog',
                          yscale: str = 'log',
                          size: Tuple[float] = (18.5, 10.5)) -> plt.Axes:
    """
    Scatter plot of function values and gradient values at the end of
    optimization. Optimizer exit-message is encoded by color. Can help
    identifying convergence issues in optimization and guide tolerance
    refinement etc.

    Parameters
    ----------

    result:
        Optimization result obtained by 'optimize.py'

    ax:
        Axes object to use.

    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    xscale:
        Scale for x-axis

    yscale:
        Scale for y-axis

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """
    if ax is None:
        ax = plt.subplots(figsize=size)[1]

    fvals = result.optimize_result.get_for_key('fval')
    grad_norms = [
        np.linalg.norm(
            result.problem.get_reduced_vector(grad,
                                              result.problem.x_free_indices),
            2
        )
        if grad is not None else np.NaN
        for grad in result.optimize_result.get_for_key('grad')
    ]
    msgs = result.optimize_result.get_for_key('message')
    conv_data = pd.DataFrame({
        'fval': fvals,
        'gradient norm': grad_norms,
        'exit message': msgs
    })
    sns.scatterplot(x='fval', y='gradient norm', hue='exit message',
                    data=conv_data, ax=ax)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    return ax
