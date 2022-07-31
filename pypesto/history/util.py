import numbers
from typing import Any, Callable, Dict, Sequence, Union

import numpy as np

from ..C import SUFFIXES

ResultDict = Dict[str, Union[float, np.ndarray]]
MaybeArray = Union[np.ndarray, 'np.nan']


class HistoryTypeError(ValueError):
    """Error raised when an unsupported history type is requested."""

    def __init__(self, history_type: str):
        super().__init__(
            f"Unsupported history type: {history_type}, expected {SUFFIXES}"
        )


class CsvHistoryTemplateError(ValueError):
    """Error raised when no template is given for CSV history."""

    def __init__(self, storage_file: str):
        super().__init__(
            "CSV History requires an `{id}` template in the `storage_file`, "
            f"but is {storage_file}"
        )


def trace_wrap(f):
    """
    Wrap around trace getters.

    Transform input `ix` vectors to a valid index list, and reduce for
    integer `ix` the output to a single value.
    """

    def wrapped_f(
        self, ix: Union[Sequence[int], int, None] = None, trim: bool = False
    ) -> Union[Sequence[Union[float, MaybeArray]], Union[float, MaybeArray]]:
        # whether to reduce the output
        reduce = isinstance(ix, numbers.Integral)
        # default: full list
        if ix is None:
            if trim:
                ix = self.get_trimmed_indices()
            else:
                ix = np.arange(0, len(self), dtype=int)
        # turn every input into an index list
        if reduce:
            ix = np.array([ix], dtype=int)
        # obtain the trace
        trace = f(self, ix)
        # reduce the output
        if reduce:
            trace = trace[0]
        return trace

    return wrapped_f


def _check_none(fun: Callable[..., Any]) -> Callable[..., Union[Any, None]]:
    """Return None if any input argument is None; Wrapper function."""

    def checked_fun(*args, **kwargs):
        if any(x is None for x in [*args, *(kwargs.values())]):
            return None
        return fun(*args, **kwargs)

    return checked_fun


@_check_none
def res_to_chi2(res: np.ndarray) -> float:
    """Translate residuals to chi2 values, `chi2 = sum(res**2)`."""
    return float(np.dot(res, res))


@_check_none
def chi2_to_fval(chi2: float) -> float:
    """Translate chi2 to function value, `fval = 0.5*chi2 = 0.5*sum(res**2)`.

    Note that for the function value we thus employ a probabilistic
    interpretation, as the log-likelihood of a standard normal noise model.
    This is in line with e.g. AMICI's and SciPy's objective definition.
    """
    return 0.5 * chi2


@_check_none
def res_to_fval(res: np.ndarray) -> float:
    """Translate residuals to function value, `fval = 0.5*sum(res**2)`."""
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
