from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..result import Result


def optimizer_convergence(
    result: Result,
    ax: Optional[plt.Axes] = None,
    xscale: str = "symlog",
    yscale: str = "log",
    size: tuple[float] = (18.5, 10.5),
) -> plt.Axes:
    """
    Visualize to help spotting convergence issues.

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
    import seaborn as sns

    if ax is None:
        ax = plt.subplots(figsize=size)[1]

    fvals = result.optimize_result.fval
    grad_norms = [
        (
            np.linalg.norm(
                result.problem.get_reduced_vector(
                    grad, result.problem.x_free_indices
                ),
                2,
            )
            if grad is not None
            else np.nan
        )
        for grad in result.optimize_result.grad
    ]
    msgs = result.optimize_result.message
    conv_data = pd.DataFrame(
        {"fval": fvals, "gradient norm": grad_norms, "exit message": msgs}
    )
    sns.scatterplot(
        x="fval", y="gradient norm", hue="exit message", data=conv_data, ax=ax
    )
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    return ax
