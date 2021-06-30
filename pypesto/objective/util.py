import numpy as np
from typing import Union


def _check_none(fun):
    """Wrapper: Return None if any input argument is None."""
    def checked_fun(*args, **kwargs):
        if any(x is None for x in [*args, *(kwargs.values())]):
            return None
        return fun(*args, **kwargs)
    return checked_fun


@_check_none
def res_to_chi2(res: np.ndarray) -> Union[float, None]:
    """Translate residuals to chi2 values, `chi2 = sum(res**2)`."""
    return np.dot(res, res)


@_check_none
def chi2_to_fval(chi2: float) -> Union[float, None]:
    """Translate chi2 to function value, `fval = 0.5*chi2 = 0.5*sum(res**2)`.

    Note that for the function value we thus employ a probabilistic
    interpretation, as the log-likelihood of a standard normal noise model.
    This is in line with e.g. AMICI's and SciPy's objective definition.
    """
    return 0.5 * chi2


@_check_none
def res_to_fval(res: np.ndarray) -> Union[float, None]:
    """Translate residuals to function value, `fval = 0.5*sum(res**2)`."""
    return chi2_to_fval(res_to_chi2(res))


@_check_none
def sres_to_schi2(res: np.ndarray, sres: np.ndarray):
    """Translate residual sensitivities to chi2 gradient."""
    return 2 * res.dot(sres)


@_check_none
def schi2_to_grad(schi2: np.ndarray) -> np.ndarray:
    """Translate chi2 gradient to function value gradient.

    See also :func:`chi2_to_fval`.
    """
    return 0.5 * schi2


@_check_none
def sres_to_grad(res: np.ndarray, sres: np.ndarray):
    """Translate residual sensitivities to function value gradien, assuming
    `fval = 0.5*sum(res**2)`.

    See also :func:`chi2_to_fval`.
     """
    return schi2_to_grad(sres_to_schi2(res, sres))


@_check_none
def sres_to_fim(sres: np.ndarray):
    """Translate residual sensitivities to FIM.

    The FIM is based on the function values, not chi2, i.e. has a normalization
    of 0.5 as in :func:`res_to_fval`.
    """
    return sres.transpose().dot(sres)
