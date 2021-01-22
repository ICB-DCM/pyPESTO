import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from ..collections import Collection
from ..collections.constants import (
    COLOR_HIT_BOTH_BOUNDS, COLOR_HIT_ONE_BOUND, COLOR_HIT_NO_BOUNDS)
from .reference_points import create_references, ReferencePoint
from .clust_color import assign_colors
from .clust_color import delete_nan_inf
from .misc import process_result_list, process_start_indices


def collection_identifiability(collection: Collection):
    id_df = collection.check_identifiability()
    none_hit, lb_hit, ub_hit, both_hit = _prepare_identifiability_plot(id_df)
    n_par = id_df.shape[0]
    n_both = len(both_hit) / n_par
    n_lb = len(lb_hit) / n_par
    n_ub = len(ub_hit) / n_par

    ax = plt.subplot(111)
    patches_both_hit = []
    x = 0.
    h = 1. / n_par
    for hit in both_hit:
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


    ax.add_collection(patches_both_hit)
    ax.add_collection(patches_lb_hit)
    ax.add_collection(patches_ub_hit)
    ax.add_collection(patches_none_hit)

    tmp = n_both
    ax.plot([tmp, tmp], [0, 1], 'k--', linewidth=1.)
    tmp += n_lb
    ax.plot([tmp, tmp], [0, 1], 'k--', linewidth=1.)
    tmp += n_ub
    ax.plot([tmp, tmp], [0, 1], 'k--', linewidth=1.)
    ax.plot([0, 0], [0, 1], 'k-', linewidth=1.)
    ax.plot([1, 1], [0, 1], 'k-', linewidth=1.)

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

