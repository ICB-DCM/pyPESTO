from collections.abc import Sequence
from typing import Literal

import matplotlib.axes
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle

from ..profile import calculate_approximate_ci, chi2_quantile_to_ratio
from ..result import Result

# kwargs passed to `matplotlib.axes.Axes.errorbar` for plotting confidence levels
cis_visualization_settings = {
    "capsize": 5,
    "linewidth": 2,
}


def profile_cis(
    result: Result,
    confidence_level: float = 0.95,
    df: int = 1,
    profile_indices: Sequence[int] = None,
    profile_list: int = 0,
    color: str | tuple = "C0",
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
    df:
        Degrees of freedom of the chi2 distribution.
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

    confidence_ratio = chi2_quantile_to_ratio(confidence_level, df=df)

    # calculate intervals
    intervals = []
    for i_par in range(problem.dim_full):
        if i_par not in profile_indices:
            continue
        xs = profile_list[i_par].x_path[i_par]
        ratios = profile_list[i_par].ratio_path
        lb, ub = calculate_approximate_ci(
            xs=xs, ratios=ratios, confidence_ratio=confidence_ratio
        )
        intervals.append((lb, ub))

    x_names = [problem.x_names[ix] for ix in profile_indices]

    for ix, (lb, ub) in enumerate(intervals):
        half = (ub - lb) / 2
        ax.errorbar(
            lb + half,
            ix + 1,
            xerr=half,
            color=color,
            **cis_visualization_settings,
        )

    parameters_ind = np.arange(1, len(intervals) + 1)
    ax.set_yticks(parameters_ind)
    ax.set_yticklabels(x_names)
    ax.set_ylabel("Parameter")
    ax.set_xlabel("Parameter value")

    if show_bounds:
        lb = problem.lb_full[profile_indices]
        ax.plot(lb, parameters_ind, "k--", marker="+")
        ub = problem.ub_full[profile_indices]
        ax.plot(ub, parameters_ind, "k--", marker="+")

    return ax


def profile_nested_cis(
    result: Result,
    confidence_levels: Sequence[float] = (0.95, 0.9),
    df: int = 1,
    profile_indices: Sequence[int] = None,
    profile_list: int = 0,
    colors: Sequence = None,
    ax: matplotlib.axes.Axes = None,
    orientation: Literal["v", "h"] = "v",
):
    """
    Plot approximate nested confidence intervals based on profiles.

    Parameters
    ----------
    result:
        The result object with profiling results.
    confidence_levels:
        The confidence levels in (0,1), which are translated to an approximate
        threshold assuming a chi2 distribution, using
        `pypesto.profile.chi2_quantile_to_ratio`.
    df:
        Degrees of freedom of the chi2 distribution.
    profile_indices:
        List of integer values specifying which profiles should be plotted.
        Defaults to the indices for which profiles were generated in profile
        list `profile_list`.
    profile_list:
        Index of the profile list to be used.
    colors:
        A color for each confidence interval.
    ax:
        Axes object to use. Default: Create a new one.
    orientation:
        Orientation of the plot, either vertical or horizontal.
    """
    # extract problem
    problem = result.problem
    # extract profile list
    profile_list = result.profile_result.list[profile_list]

    n_cls = len(confidence_levels)
    ws = [(0.6 / n_cls) * i for i in range(1, n_cls + 1)]
    if colors is None:
        blues = cm.get_cmap("Blues")
        colors = [blues(i) for i in ws]

    # ensure that the confidence levels are sorted in decreasing order
    confidence_levels, colors = zip(
        *sorted(zip(confidence_levels, colors, strict=True), reverse=True),
        strict=True,
    )

    if profile_indices is None:
        profile_indices = [ix for ix, res in enumerate(profile_list) if res]

    if ax is None:
        _, ax = plt.subplots()

    legends = []
    for i, confidence_level in enumerate(confidence_levels):
        confidence_ratio = chi2_quantile_to_ratio(confidence_level, df=df)

        xs_list = []
        x = -ws[i] / 2
        rectangles = []
        for j, i_par in enumerate(profile_indices):
            conf_l_indices = [
                idx
                for idx, ratio in enumerate(profile_list[i_par].ratio_path)
                if ratio >= confidence_ratio
            ]
            xs = profile_list[i_par].x_path[i_par][conf_l_indices]
            xs_list.append(xs)

            par_ci = [np.min(xs), np.max(xs)]
            h = par_ci[1] - par_ci[0]

            if orientation == "v":
                rectangles.append(Rectangle((par_ci[0], x), h, ws[i]))
            else:
                rectangles.append(Rectangle((x, par_ci[0]), ws[i], h))
            x += 1

            # visualize parameter boundaries
            if orientation == "v":
                ax.plot(
                    [result.problem.lb_full[i_par]] * 2,
                    [j - 0.4, j + 0.4],
                    color="grey",
                )
                ax.plot(
                    [result.problem.ub_full[i_par]] * 2,
                    [j - 0.4, j + 0.4],
                    color="grey",
                )
            else:
                ax.plot(
                    [j - 0.4, j + 0.4],
                    [result.problem.lb_full[i_par]] * 2,
                    color="grey",
                )
                ax.plot(
                    [j - 0.4, j + 0.4],
                    [result.problem.ub_full[i_par]] * 2,
                    color="grey",
                )

        ax.add_collection(
            PatchCollection(
                rectangles, facecolors=colors[i], edgecolors="dimgrey"
            )
        )
        legends.append(
            Patch(color=colors[i], label=f"{confidence_level * 100}%")
        )

    x_names = [problem.x_names[ix] for ix in profile_indices]
    parameters_ind = np.arange(0, len(profile_indices))

    if orientation == "v":
        ax.set_yticks(parameters_ind)
        ax.set_yticklabels(x_names)
        ax.set_ylabel("Parameter")
        ax.set_xlabel("Parameter value")
    else:
        ax.set_xticks(parameters_ind)
        ax.set_xticklabels(ax.get_xticklabels(), ha="right")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Parameter value")

    ax.legend(
        title="Confidence level:",
        handles=legends,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=len(legends),
    )

    return ax
