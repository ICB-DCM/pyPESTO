import numpy as np


def res_to_chi2(res):
    """
    We assume that the residuals res are related to an objective function
    value fval = chi2 via::
        fval = 0.5 * sum(res**2)
    which is the 'Linear' formulation in scipy.

    """
    if res is None:
        return None
    return 0.5 * np.power(res, 2).sum()


def sres_to_schi2(res, sres):
    """
    In line with the assumptions in res_to_chi2.

    """
    if res is None or sres is None:
        return None
    return res * sres
