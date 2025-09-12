"""
Utilities
=========

Package-wide utilities.

"""

from collections.abc import Sequence
from numbers import Number
from operator import itemgetter
from typing import Any, Callable, Optional, Union

import numpy as np
from tqdm import tqdm as _tqdm


def _check_none(fun: Callable[..., Any]) -> Callable[..., Union[Any, None]]:
    """Return None if any input argument is None; Wrapper function."""

    def checked_fun(*args, **kwargs):
        if any(x is None for x in [*args, *(kwargs.values())]):
            return None
        return fun(*args, **kwargs)

    return checked_fun


@_check_none
def res_to_chi2(res: np.ndarray) -> float:
    """Translate residuals to chi2 values, `chi2 = sum(res**2) + C`."""
    return float(np.dot(res, res))


@_check_none
def chi2_to_fval(chi2: float) -> float:
    """Translate chi2 to function value, `fval = 0.5*chi2 = 0.5*sum(res**2) + C`.

    Note that for the function value we thus employ a probabilistic
    interpretation, as the log-likelihood of a standard normal noise model.
    This is in line with e.g. AMICI's and SciPy's objective definition.
    """
    return 0.5 * chi2


@_check_none
def fval_to_chi2(fval: float) -> float:
    """Translate function value to chi2, `chi2 = 2 * fval`.

    Note that for the function value we thus employ a probabilistic
    interpretation, as the log-likelihood of a standard normal noise model.
    This is in line with e.g. AMICI's and SciPy's objective definition.
    """
    return 2.0 * fval


@_check_none
def res_to_fval(res: np.ndarray) -> float:
    """Translate residuals to function value, `fval = 0.5*sum(res**2) + C`."""
    return chi2_to_fval(res_to_chi2(res))


@_check_none
def sres_to_schi2(res: np.ndarray, sres: np.ndarray) -> np.ndarray:
    """Translate residual sensitivities to chi2 gradient."""
    return 2 * res.dot(sres)


@_check_none
def schi2_to_grad(schi2: np.ndarray) -> np.ndarray:
    """Translate chi2 gradient to function value gradient.

    See also :func:`chi2_to_fval`.
    """
    return 0.5 * schi2


@_check_none
def grad_to_schi2(grad: np.ndarray) -> np.ndarray:
    """Translate function value gradient to chi2 gradient."""
    return 2.0 * grad


@_check_none
def sres_to_grad(res: np.ndarray, sres: np.ndarray) -> np.ndarray:
    """Translate residual sensitivities to function value gradient.

    Assumes `fval = 0.5*sum(res**2)`.

    See also :func:`chi2_to_fval`.
    """
    return schi2_to_grad(sres_to_schi2(res, sres))


@_check_none
def sres_to_fim(sres: np.ndarray) -> np.ndarray:
    """Translate residual sensitivities to FIM.

    The FIM is based on the function values, not chi2, i.e. has a normalization
    of 0.5 as in :func:`res_to_fval`.
    """
    return sres.transpose().dot(sres)


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
    return f"condition_{condition_id}"


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

    # sort values
    vals = np.sort(vals)

    # assign cluster to first element and then assign new cluster every time distance is greater 0.1
    cluster_indices = np.append([0], np.cumsum(np.diff(vals) > 0.1))

    # get cluster sizes
    _, cluster_size = np.unique(cluster_indices, return_counts=True)
    return cluster_indices, cluster_size


def delete_nan_inf(
    fvals: np.ndarray,
    x: Optional[Sequence[Union[np.ndarray, list[float]]]] = None,
    xdim: Optional[int] = 1,
    magnitude_bound: Optional[float] = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
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
    magnitude_bound:
        any values with a magnitude (absolute value) larger than the
        `magnitude_bound` are also deleted

    Returns
    -------
    x:
        array of parameters without nan or inf
    fvals:
        array of fval without nan or inf
    """
    fvals = np.asarray(fvals)
    finite_fvals = np.isfinite(fvals) & (np.absolute(fvals) < magnitude_bound)
    if x is not None:
        if np.isfinite(fvals).any():
            finite_indices = np.where(finite_fvals)[0]
            x = np.array(itemgetter(*finite_indices)(x))
            # itemgetter does not return list for single element
            if len(x.shape) == 1:
                x = x.reshape((1, x.shape[0]))
        else:
            # If none of the fvals are finite we try to construct an
            # empty np.ndarray with the correct dimension (other functions rely
            # on x.shape[1] to be of correct dimension)
            x = np.array(x)
            x = np.empty(
                (
                    0,
                    (
                        x.shape[1]
                        if x.ndim == 2
                        else x[0].shape[0]
                        if x[0] is not None
                        else xdim
                    ),
                )
            )
    return x, fvals[finite_fvals]


def tqdm(*args, enable: bool = None, **kwargs):
    """
    Create a progress bar using tqdm.

    Parameters
    ----------
    args:
        Arguments passed to tqdm.
    enable:
        Whether to enable the progress bar.
        If None, use tqdm defaults.
        Mutually exclusive with `disable`.
    kwargs:
        Keyword arguments passed to tqdm.

    Returns
    -------
    progress_bar:
        A progress bar.
    """
    # Drop the `disable` argument unless it is not-None.
    # This way, we don't interfere with TQDM_DISABLE or other global
    # tqdm settings.
    disable = kwargs.pop("disable", None)

    if enable is not None:
        if disable is not None and enable != disable:
            raise ValueError(
                "Contradicting values for `enable` and `disable` passed."
            )
        disable = not enable

    if disable is not None:
        kwargs["disable"] = disable
    return _tqdm(*args, **kwargs)
