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

    Attributes
    ----------

    objective: pypesto.Objective
        The objective function for minimization.

    lb, ub: array_like
        The lower and upper bounds. For unbounded directions set to inf. 

    dim_full: int, optional
        The full dimension of the problem, including fixed parameters.

    dim: int
        The number of non-fixed parameters.
        Computed from the other values.
        
    x_fixed_indices: array_like of int, optional
        Vector containing the indices (zero-based) of parameter components
        that are not to be optipmized.

    x_fixed_vals: array_like, optional
        Vector of the same length as x_fixed_indices, containing the values
        of the fixed parameters.

    x_guesses: array_like, optional
        Guesses for the parameter values, shape (g, dim), where g denotes the
        number of guesses. These are used as start points in the optimization.

    Notes
    -----

    On the fixing of parameter values:

    The number of parameters dim_full the objective takes as input must
    be known, so it must be either lb a vector of that size, or dim_full 
    specified as a parameter. 
    All vectors are mapped to the reduced space of dimension dim in __init__,
    regardless of whether they were in dimension dim or dim_full before.
    """

    def __init__(self,
                 objective,
                 lb, ub,
                 dim_full=None,
                 x_fixed_indices=None,
                 x_fixed_vals=None,
                 x_guesses=None):

        self.objective = objective

        self.lb = np.array(lb).flatten()
        self.ub = np.array(ub).flatten()

        self.dim_full = dim_full if dim_full is not None else self.lb.size
        
        if x_fixed_indices is None:
            x_fixed_indices = np.array([])
        self.x_fixed_indices = np.array(x_fixed_indices)

        if x_fixed_vals is None:
            x_fixed_vals = np.array([])
        self.x_fixed_vals = np.array(x_fixed_vals)

        self.dim = self.dim_full - self.x_fixed_indices.size

        self.x_free_indices = np.array(
            list(set(range(0, self.dim_full)) - set(self.x_fixed_vals))
        )

        if x_guesses is None:
            x_guesses = np.zeros((0, self.dim))
        self.x_guesses = np.array(x_guesses)

        self.normalize_input()

    def normalize_input(self):
        """
        Reduce all vectors to dimension dim and have the objective accept
        vectors of dimension dim.
        """

        if self.lb.size == self.dim_full:
            self.lb = self.lb[self.x_free_indices]
        elif self.lb.size == 1:
            self.lb = self.lb * np.ones(self.dim)

        if self.ub.size == self.dim_full:
            self.ub = self.ub[self.x_free_indices]
        elif self.ub.size == 1:
            self.ub = self.ub * np.ones(self.dim)

        if self.x_guesses.shape[1] == self.dim_full:
            self.x_guesses = self.x_guesses[:, self.x_free_indices]

        assert self.lb.size == self.dim
        assert self.ub.size == self.dim
        assert self.x_guesses.shape[1] == self.dim
