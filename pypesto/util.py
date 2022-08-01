"""
Utilities
=========

Package-wide utilities.

"""
from numbers import Number
from typing import Optional, Tuple, Union

import numpy as np
from scipy import cluster


def is_none_or_nan(x: Union[Number, None]) -> bool:
    """
    Check if x is None or NaN.

    Parameters
    ----------
    x:
        object to be checked

    Returns
    -------
    True if x is None or NaN, False otherwise.
    """
    return x is None or np.isnan(x)


def is_none_or_nan_array(x: Union[Number, np.ndarray, None]) -> bool:
    """
    Check if x is None or NaN array.

    Parameters
    ----------
    x:
        object to be checked

    Returns
    -------
    True if x is None or NaN array, False otherwise.
    """
    return x is None or np.isnan(x).all()


def allclose(
    x: Union[Number, np.ndarray], y: Union[Number, np.ndarray]
) -> bool:
    """
    Check if two arrays are close.

    Parameters
    ----------
    x: first array
    y: second array

    Returns
    -------
    True if all elements of x and y are close, False otherwise.
    """
    # Note: We use this wrapper around np.allclose in order to more easily
    #  adjust hyper parameters for the tolerance.
    return np.allclose(x, y)


def isclose(
    x: Union[Number, np.ndarray],
    y: Union[Number, np.ndarray],
) -> Union[bool, np.ndarray]:
    """
    Check if two values or arrays are close, element-wise.

    Parameters
    ----------
    x: first array
    y: second array

    Returns
    -------
    Element-wise boolean comparison of x and y.
    """
    # Note: We use this wrapper around np.isclose in order to more easily
    #  adjust hyper parameters for the tolerance.
    return np.isclose(x, y)


def get_condition_label(condition_id: str) -> str:
    """Convert a condition ID to a label.

    Labels for conditions are used at different locations (e.g. ensemble
    prediction code, and visualization code). This method ensures that the same
    condition is labeled identically everywhere.

    Parameters
    ----------
    condition_id:
        The condition ID that will be used to generate a label.

    Returns
    -------
    The condition label.
    """
    return f'condition_{condition_id}'


def assign_clusters(vals):
    """
    Find clustering.

    Parameters
    ----------
    vals: numeric list or array
        List to be clustered.

    Returns
    -------
    clust: numeric list
        Indicating the corresponding cluster of each element from
        'vals'.
    clustsize: numeric list
        Size of clusters, length equals number of clusters.
    """
    # sanity checks
    if vals is None or len(vals) == 0:
        return [], []
    elif len(vals) == 1:
        return np.array([0]), np.array([1.0])

    # linkage requires (n, 1) data array
    vals = np.reshape(vals, (-1, 1))

    # however: clusters are sorted by size, not by value... Resort.
    # Create preallocated object first
    cluster_indices = np.zeros(vals.size, dtype=int)

    # get clustering based on distance
    clust = cluster.hierarchy.fcluster(
        cluster.hierarchy.linkage(vals), t=0.1, criterion='distance'
    )

    # get unique clusters
    _, ind_clust = np.unique(clust, return_index=True)
    unique_clust = clust[np.sort(ind_clust)]
    cluster_size = np.zeros(unique_clust.size, dtype=int)

    # loop over clusters: resort and count number of entries
    for index, i_clust in enumerate(unique_clust):
        cluster_indices[np.where(clust == i_clust)] = index
        cluster_size[index] = sum(clust == i_clust)

    return cluster_indices, cluster_size


def delete_nan_inf(
    fvals: np.ndarray, x: Optional[np.ndarray] = None, xdim: Optional[int] = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Delete nan and inf values in fvals.

    If parameters 'x' are passed, also the corresponding entries are deleted.

    Parameters
    ----------
    x:
        array of parameters
    fvals:
        array of fval
    xdim:
        dimension of x, in case x dimension cannot be inferred

    Returns
    -------
    x:
        array of parameters without nan or inf
    fvals:
        array of fval without nan or inf
    """
    fvals = np.asarray(fvals)
    if x is not None:
        # if we start out with a list of x, the x corresponding
        # to finite fvals may be None, so we cannot stack the x before taking
        # subindexing
        # If none of the fvals are finite, np.vstack will fail and np.take
        # will not yield the correct dimension, so we try to construct an
        # empty np.ndarray with the correct dimension (other functions rely
        # on x.shape[1] to be of correct dimension)
        if np.isfinite(fvals).any():
            x = np.vstack(np.take(x, np.where(np.isfinite(fvals))[0], axis=0))
        else:
            x = np.empty(
                (
                    0,
                    x.shape[1]
                    if x.ndim == 2
                    else x[0].shape[0]
                    if x[0] is not None
                    else xdim,
                )
            )
    return x, fvals[np.isfinite(fvals)]
