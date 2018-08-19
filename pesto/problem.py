"""
Problem
-------

A problem contains the objective as well as all information like prior
describing the problem to be solved.

"""


import numpy as np


class Problem:
    """
    The problem formulation. A problem specifies the objective function,
    boundaries and constraints, parameter guesses as well as the parameters
    which are to be optimized.

    Parameters
    ----------

    objective: pesto.Objective
        The objective function for minimization.

    lb, ub: array_like
        The lower and upper bounds. For unbounded problems set to inf.

    par_guesses: array_like
        Guesses for the parameter values, shape (dim, g) where g denotes the
        number of guesses. These are used as start points in the optimization.

    """

    def __init__(self,
                 objective,
                 lb, ub,
                 par_guesses=None):
        self.objective = objective
        self.lb = np.asarray(lb).reshape((1, -1))
        self.ub = np.asarray(ub).reshape((1, -1))
        self.dim = self.lb.shape[1]
        self.par_guesses = par_guesses
