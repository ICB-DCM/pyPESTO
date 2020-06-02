import matplotlib.axes
import matplotlib.pyplot as plt
from typing import Sequence, Union
import numpy as np

from ..result import Result
from ..profile import chi2_quantile_to_ratio, calculate_approximate_ci


def profile_cis(
        result: Result,
        confidence_level: float = 0.95,
        profile_indices: Sequence[int] = None,
        profile_list: int = 0,
        color: Union[str, tuple] = 'C0',
        show_bounds: bool = False,
        ax: matplotlib.axes.Axes = None,
) -> matplotlib.axes.Axes:
    """
    Plot approximate confidence intervals based on profiles.

    Parameters
    ----------
    result:
        The result object after profiling.
    confidence_level:
        The confidence level in (0,1), which is translated to an approximate
        threshold assuming a chi2 distribution, using
        `pypesto.profile.chi2_quantile_to_ratio`.
    profile_indices:
        List of integer values specifying which profiles should be plotted.
        Defaults to the indices for which profiles were generated in profile
        list `profile_list`.
    profile_list:
        Index of the profile list to be used.
    color:
        Main plot color.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.
    ax:
        Axes object to use. Default: Create a new one.
    """
    # extract problem
    problem = result.problem
    # extract profile list
    profile_list = result.profile_result.list[profile_list]

    if profile_indices is None:
        profile_indices = [ix for ix, res in enumerate(profile_list) if res]

    if ax is None:
        _, ax = plt.subplots()

    confidence_ratio = chi2_quantile_to_ratio(confidence_level)

    # calculate intervals
    intervals = []
    for i_par in range(problem.dim_full):
        if i_par not in profile_indices:
            continue
        xs = profile_list[i_par].x_path[i_par]
        ratios = profile_list[i_par].ratio_path
        lb, ub = calculate_approximate_ci(
            xs=xs, ratios=ratios, confidence_ratio=confidence_ratio)
        intervals.append((lb, ub))

    x_names = [problem.x_names[ix] for ix in profile_indices]

    for ix, (lb, ub) in enumerate(intervals):
        ax.plot([lb, ub], [ix+1, ix+1], marker='|', color=color)

    parameters_ind = np.arange(1, len(intervals) + 1)
    ax.set_yticks(parameters_ind)
    ax.set_yticklabels(x_names)
    ax.set_ylabel("Parameter")
    ax.set_xlabel("Parameter value")

    if show_bounds:
        lb = problem.lb_full[profile_indices]
        ax.plot(lb, parameters_ind, 'k--', marker='+')
        ub = problem.ub_full[profile_indices]
        ax.plot(ub, parameters_ind, 'k--', marker='+')

    return ax
