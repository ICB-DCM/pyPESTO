import numpy as np


def res_to_fval(res):
    """
    We assume that the residuals res are related to an objective function
    value fval via::
        fval = 0.5 * sum(res**2),
    which is the 'Linear' formulation in scipy.
    """
    if res is None:
        return None
    return 0.5 * np.power(res, 2).sum()
