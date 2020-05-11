import numpy as np


def res_to_chi2(res: np.ndarray):
    """
    We assume that the residuals res are related to an objective function
    value chi2 via::

        chi2 = sum(res**2)

    which is consistent with the AMICI definition but NOT the 'Linear'
    formulation in scipy.
    """
    if res is None:
        return None
    return np.dot(res, res)


def sres_to_schi2(res: np.ndarray, sres: np.ndarray):
    """
    In line with the assumptions in res_to_chi2.
    """
    if res is None or sres is None:
        return None
    return 2 * res.dot(sres)


def sres_to_fim(sres: np.ndarray):
    """
    In line with the assumptions in res_to_chi2.
    """
    if sres is None:
        return None
    return sres.transpose().dot(sres)
