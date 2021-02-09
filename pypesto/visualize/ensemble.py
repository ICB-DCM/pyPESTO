import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from typing import Optional, Tuple

from ..ensemble import Ensemble
from ..ensemble.constants import (
    COLOR_HIT_BOTH_BOUNDS, COLOR_HIT_ONE_BOUND, COLOR_HIT_NO_BOUNDS)


def ensemble_identifiability(ensemble: Ensemble,
                             ax: Optional[plt.Axes] = None,
                             size: Optional[Tuple[float]] = (12, 6)):
    """
    Plots an overview about how many parameters hit the parameter bounds based
    on a ensemble of parameters. confidence intervals/credible ranges are
    computed via the ensemble mean plus/minus 1 standard deviation.
    This highlevel routine expects a ensemble object as input.

    Parameters
    ----------

    ensemble:
        ensemble of parameter vectors (from pypesto.ensemble)

    ax:
        Axes object to use.

    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # first get the data to check identifiability
    id_df = ensemble.check_identifiability()

    # check how many bounds are actually hit and which ones
    none_hit, lb_hit, ub_hit, both_hit = _prepare_identifiability_plot(id_df)

    # call lowlevel routine whick works with np arrays only
    ax = ensemble_identifiability_lowlevel(none_hit, lb_hit, ub_hit, both_hit,
                                           ax, size)

    return ax


def ensemble_identifiability_lowlevel(none_hit: np.ndarray,
                                      lb_hit: np.ndarray,
                                      ub_hit: np.ndarray,
                                      both_hit: np.ndarray,
                                      ax: Optional[plt.Axes] = None,
                                      size: Optional[Tuple[float]] = (16, 10)):
    """
    Plots an overview about how many parameters hit the parameter bounds based
    on a ensemble of parameters. Confidence intervals/credible ranges are
    computed via the ensemble mean plus/minus 1 standard deviation.
    This lowlevel routine works with numpy arrays which define the confidence
    intervals/credible ranges of each parameter.

    Parameters
    ----------

    none_hit:
        2-dimensional array of confidence interval/credible ranges for
        identifiable parameters

    lb_hit:
        2-dimensional array of confidence interval/credible ranges for
        parameters which hit the lower parameter bound

    ub_hit:
        2-dimensional array of confidence interval/credible ranges for
        parameters which hit the upper parameter bound

    both_hit:
        2-dimensional array of confidence interval/credible ranges for
        parameters which hit both parameter bounds

    ax:
        Axes object to use.

    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # define some short hands for later plotting
    n_par = sum([none_hit.shape[0], lb_hit.shape[0],
                 ub_hit.shape[0], both_hit.shape[0]])
    x_both = len(both_hit) / n_par
    x_lb = len(lb_hit) / n_par
    x_ub = len(ub_hit) / n_par
    x_none = 1. - x_both - x_ub - x_lb

    patches_both_hit, patches_lb_hit, patches_ub_hit, patches_none_hit = \
        _create_patches(none_hit, lb_hit, ub_hit, both_hit)

    # axes
    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # create axes object and add patch collections
    if patches_both_hit:
        ax.add_collection(patches_both_hit)
    if patches_lb_hit:
        ax.add_collection(patches_lb_hit)
    if patches_ub_hit:
        ax.add_collection(patches_ub_hit)
    if patches_none_hit:
        ax.add_collection(patches_none_hit)

    # plot dashed lines indicating the number rof non-identifiable parameters
    vert = [-.05, 1.05]
    ax.plot([x_both, x_both], vert, 'k--', linewidth=1.5)
    ax.plot([x_both + x_lb, x_both + x_lb], vert, 'k--', linewidth=1.5)
    ax.plot([x_both + x_lb + x_ub, x_both + x_lb + x_ub], vert,
            'k--', linewidth=1.5)

    # add text
    if patches_both_hit:
        ax.text(x_both / 2, -.05, 'both bounds hit',
                color=COLOR_HIT_BOTH_BOUNDS,
                rotation=-90, va='top', ha='center')
    if patches_lb_hit:
        ax.text(x_both + x_lb / 2, -.05, 'lower bound hit',
                color=COLOR_HIT_ONE_BOUND, rotation=-90, va='top', ha='center')
    if patches_ub_hit:
        ax.text(x_both + x_lb + x_ub / 2, -.05, 'upper bound hit',
                color=COLOR_HIT_ONE_BOUND, rotation=-90, va='top', ha='center')
    if patches_none_hit:
        ax.text(1 - x_none / 2, -.05, 'no bounds hit',
                color=COLOR_HIT_NO_BOUNDS, rotation=-90, va='top', ha='center')
    ax.text(0, -.7, 'identifiable parameters: {:4.1f}%'.format(x_none * 100),
            va='top')

    # plot upper and lower bounds
    ax.text(-.03, 1., 'upper\nbound', ha='right', va='center')
    ax.text(-.03, 0., 'lower\nbound', ha='right', va='center')
    ax.plot([-.02, 1.03], [0, 0], 'k:', linewidth=1.5)
    ax.plot([-.02, 1.03], [1, 1], 'k:', linewidth=1.5)
    plt.xticks([])
    plt.yticks([])

    # plot frame
    ax.plot([0, 0], vert, 'k-', linewidth=1.5)
    ax.plot([1, 1], vert, 'k-', linewidth=1.5)

    # beautify axes
    plt.xlim((-.15, 1.1))
    plt.ylim((-.78, 1.15))
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax


def _prepare_identifiability_plot(id_df: pd.DataFrame):
    """
    This routine groups model parameters based on a ensemble object into
    four categories, based on the mean of the parameter ensemble plus/minus
    1 standard deviation: Parameters that hit both bounds, parameters that hit
    only the lower [or upper] bound, and parameters that hit no bounds.
    It returns them as four numpy arrays, together with their confidence
    intervals/credible ranges.

    Parameters
    ----------
    id_df:
        Pandas dataframe with information about parameter identifiability,
        as created by pypesto.ensemble.check_identifiability()

    Returns
    -------
    none_hit:
        2-dimensional array of confidence interval/credible ranges for
        identifiable parameters

    lb_hit:
        2-dimensional array of confidence interval/credible ranges for
        parameters which hit the lower parameter bound

    ub_hit:
        2-dimensional array of confidence interval/credible ranges for
        parameters which hit the upper parameter bound

    both_hit:
        2-dimensional array of confidence interval/credible ranges for
        parameters which hit both parameter bounds
    """

    # prepare
    both_hit = []
    lb_hit = []
    ub_hit = []
    none_hit = []

    def _affine_transform(par_info):
        # rescale parameters to bounds
        lb = par_info['lowerBound']
        ub = par_info['upperBound']
        val_l = par_info['ensemble_mean'] - par_info['ensemble_std']
        val_u = par_info['ensemble_mean'] + par_info['ensemble_std']
        # check if parameter confidence intervals/credible ranges hit bound
        if val_l <= lb:
            lower_val = 0.
        else:
            lower_val = (val_l - lb) / (ub - lb)
        if val_u >= ub:
            upper_val = 1.
        else:
            upper_val = (val_u - lb) / (ub - lb)

        return lower_val, upper_val

    for par_id in list(id_df.index):
        # check which of the parameters seems to be identifiable and group them
        if id_df.loc[par_id, 'within lb: 1 std'] and \
                id_df.loc[par_id, 'within ub: 1 std']:
            none_hit.append(_affine_transform(id_df.loc[par_id, :]))
        elif id_df.loc[par_id, 'within lb: 1 std']:
            ub_hit.append(_affine_transform(id_df.loc[par_id, :]))
        elif id_df.loc[par_id, 'within ub: 1 std']:
            lb_hit.append(_affine_transform(id_df.loc[par_id, :]))
        else:
            both_hit.append(_affine_transform(id_df.loc[par_id, :]))

    return np.array(none_hit), np.array(lb_hit), np.array(ub_hit), \
        np.array(both_hit)


def _create_patches(none_hit: np.ndarray,
                    lb_hit: np.ndarray,
                    ub_hit: np.ndarray,
                    both_hit: np.ndarray):
    """
    Creates matplotlib.patches.PatchCollection objects from numpy arrays with
    confidence intervals/credible ranges, which visualize identifiability
    properties of the model parameters.

    Parameters
    ----------
    none_hit:
        2-dimensional array of confidence interval/credible ranges for
        identifiable parameters

    lb_hit:
        2-dimensional array of confidence interval/credible ranges for
        parameters which hit the lower parameter bound

    ub_hit:
        2-dimensional array of confidence interval/credible ranges for
        parameters which hit the upper parameter bound

    both_hit:
        2-dimensional array of confidence interval/credible ranges for
        parameters which hit both parameter bounds

    Returns
    -------
    patches_both_hit:
        patches showing parameters which hit both parameter bounds in the
        ensemble (and are hence non-identifiable)

    patches_lb_hit:
        patches showing parameters which hit only the lower parameter bounds
        in the ensemble (and are hence non-identifiable)

    patches_ub_hit:
        patches showing parameters which hit only the lower parameter bounds
        in the ensemble (and are hence non-identifiable)

    patches_none_hit
        patches showing parameters which hit no parameter bounds in the
        ensemble (and are hence identifiable)
    """
    # get total number of parameters
    n_par = sum([none_hit.shape[0], lb_hit.shape[0],
                 ub_hit.shape[0], both_hit.shape[0]])

    # start patches at the left end and increment by h = 1/n_par
    x = 0.
    h = 1. / n_par

    # creates patches for parameters which hit both bounds
    patches_both_hit = []
    if both_hit.size > 0:
        for _ in both_hit:
            # create a list of rectangles
            patches_both_hit.append(Rectangle((x, 0.), h, 1.))
            x += h
        patches_both_hit = PatchCollection(patches_both_hit,
                                           facecolors=COLOR_HIT_BOTH_BOUNDS)

    # creates patches for parameters which hit lower bound
    patches_lb_hit = []
    # sort by normalizes length of confidence interval/credible range
    if lb_hit.size > 0:
        tmp_lb = np.sort(lb_hit[:, 1])[::-1]
        for lb_par in tmp_lb:
            # create a list of rectangles
            patches_lb_hit.append(Rectangle((x, 0.), h, lb_par))
            x += h
        patches_lb_hit = PatchCollection(patches_lb_hit,
                                         facecolors=COLOR_HIT_ONE_BOUND)

    # creates patches for parameters which hit upper bound
    patches_ub_hit = []
    # sort by normalizes length of confidence interval/credible range
    if ub_hit.size > 0:
        tmp_ub = np.sort(ub_hit[:, 0])
        for ub_par in tmp_ub:
            # create a list of rectangles
            patches_ub_hit.append(Rectangle((x, ub_par), h, 1. - ub_par))
            x += h
        patches_ub_hit = PatchCollection(patches_ub_hit,
                                         facecolors=COLOR_HIT_ONE_BOUND)

    # creates patches for parameters which hit no bounds
    patches_none_hit = []
    # sort by normalizes length of confidence interval/credible range
    if none_hit.size > 0:
        tmp_none = np.argsort(none_hit[:, 1] - none_hit[:, 0])[::-1]
        for none_par in tmp_none:
            patches_none_hit.append(
                # create a list of rectangles
                Rectangle((x, none_hit[none_par, 0]), h,
                          none_hit[none_par, 1] - none_hit[none_par, 0]))
            x += h
        patches_none_hit = PatchCollection(patches_none_hit,
                                           facecolors=COLOR_HIT_NO_BOUNDS)

    return patches_both_hit, patches_lb_hit, patches_ub_hit, patches_none_hit
