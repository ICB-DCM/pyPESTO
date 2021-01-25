import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from ..collections import Collection
from ..collections.constants import (
    COLOR_HIT_BOTH_BOUNDS, COLOR_HIT_ONE_BOUND, COLOR_HIT_NO_BOUNDS)


def collection_identifiability(collection: Collection):
    # first get the data to check identifiability
    id_df = collection.check_identifiability()

    # check how many bounds are actually hit and which ones
    none_hit, lb_hit, ub_hit, both_hit = _prepare_identifiability_plot(id_df)

    # define some short hands for later plotting
    n_par = id_df.shape[0]
    x_both = len(both_hit) / n_par
    x_lb = len(lb_hit) / n_par
    x_ub = len(ub_hit) / n_par
    x_none = 1. - x_both - x_ub - x_lb

    patches_both_hit = []
    x = 0.
    h = 1. / n_par
    for _ in both_hit:
        patches_both_hit.append(Rectangle((x, 0.), h, 1.))
        x += h
    patches_both_hit = PatchCollection(patches_both_hit,
                                       facecolors=COLOR_HIT_BOTH_BOUNDS)

    patches_lb_hit = []
    tmp_lb = np.sort(lb_hit[:,1])[::-1]
    for lb_par in tmp_lb:
        patches_lb_hit.append(Rectangle((x, 0.), h, lb_par))
        x += h
    patches_lb_hit = PatchCollection(patches_lb_hit,
                                     facecolors=COLOR_HIT_ONE_BOUND)

    patches_ub_hit = []
    tmp_ub = np.sort(ub_hit[:, 0])
    for ub_par in tmp_ub:
        patches_ub_hit.append(Rectangle((x, ub_par), h, 1. - ub_par))
        x += h
    patches_ub_hit = PatchCollection(patches_ub_hit,
                                     facecolors=COLOR_HIT_ONE_BOUND)

    patches_none_hit = []
    tmp_none = np.argsort(none_hit[:, 1] - none_hit[:, 0])[::-1]
    for none_par in tmp_none:
        patches_none_hit.append(
            Rectangle((x, none_hit[none_par, 0]), h,
                      none_hit[none_par, 1] - none_hit[none_par, 0]))
        x += h
    patches_none_hit = PatchCollection(patches_none_hit,
                                       facecolors=COLOR_HIT_NO_BOUNDS)

    # create axes object and add patch collections
    ax = plt.subplot(111)
    ax.add_collection(patches_both_hit)
    ax.add_collection(patches_lb_hit)
    ax.add_collection(patches_ub_hit)
    ax.add_collection(patches_none_hit)

    # plot dashed lines indicating the number rof non-identifiable parameters
    vert = [-.05, 1.05]
    ax.plot([x_both, x_both], vert, 'k--', linewidth=1.5)
    ax.plot([x_both + x_lb, x_both + x_lb], vert, 'k--', linewidth=1.5)
    ax.plot([x_both + x_lb + x_ub, x_both + x_lb + x_ub], vert,
            'k--', linewidth=1.5)

    # plot frame
    ax.plot([0, 0], vert, 'k-', linewidth=1.5)
    ax.plot([1, 1], vert, 'k-', linewidth=1.5)

    # plot upper and lower bounds
    ax.text(-.03, 1., 'upper\nbound', ha='right', va='center')
    ax.text(-.03, 0., 'lower\nbound', ha='right', va='center')
    ax.plot([-.02, 1.03], [0, 0], 'k:', linewidth=1.5)
    ax.plot([-.02, 1.03], [1, 1], 'k:', linewidth=1.5)
    plt.xticks([])
    plt.yticks([])
    
    ax.text(x_both / 2, -.05, 'both bounds hit', color=COLOR_HIT_BOTH_BOUNDS,
            rotation=-90, va='top', ha='center')
    ax.text(x_both + x_lb / 2, -.05, 'lower bound hit',
            color=COLOR_HIT_ONE_BOUND, rotation=-90, va='top', ha='center')
    ax.text(x_both + x_lb + x_ub / 2, -.05, 'upper bound hit',
            color=COLOR_HIT_ONE_BOUND, rotation=-90, va='top', ha='center')
    ax.text(1 - x_none / 2, -.05, 'no bounds hit',
            color=COLOR_HIT_NO_BOUNDS, rotation=-90, va='top', ha='center')
    ax.text(0, -.7, 'identifiable parameters: {:4.1f}%'.format(x_none * 100),
            va='top')

    plt.xlim((-.15, 1.1))
    plt.ylim((-.78, 1.15))
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()


def _prepare_identifiability_plot(id_df):
    both_hit = []
    lb_hit = []
    ub_hit = []
    none_hit = []

    def _affine_transform(par_info):
        lb = par_info['lowerBound']
        ub = par_info['upperBound']
        val_l = par_info['collection_mean'] - par_info['collection_std']
        val_u = par_info['collection_mean'] + par_info['collection_std']
        if val_l <= lb:
            l = 0.
        else:
            l = (val_l - lb) / (ub - lb)
        if val_u >= ub:
            u = 1.
        else:
            u = (val_u - lb) / (ub - lb)

        return l, u

    for par_id in list(id_df.index):
        # check which of the parameters seems to be identifiable
        if id_df.loc[par_id, 'within lb: 1 std'] and \
                id_df.loc[par_id, 'within ub: 1 std']:
            none_hit.append(_affine_transform(id_df.loc[par_id,:]))
        elif id_df.loc[par_id, 'within lb: 1 std']:
            ub_hit.append(_affine_transform(id_df.loc[par_id,:]))
        elif id_df.loc[par_id, 'within ub: 1 std']:
            lb_hit.append(_affine_transform(id_df.loc[par_id,:]))
        else:
            both_hit.append(_affine_transform(id_df.loc[par_id,:]))

    return np.array(none_hit), np.array(lb_hit), \
           np.array(ub_hit), np.array(both_hit)

