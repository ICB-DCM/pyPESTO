import numpy as np

class Problem:
    """
    The problem formulation.

    Parameters
    ----------

    objective: pesto.Objective
        The objective function for minimization.

    lb, ub: array_like
        The lower and upper bounds. For unbounded problems set to inf.

    par_guesses: array_like
        Guesses for the parameter values, shape (dim,g) where g denotes the
        number of guesses. These are used as start points in the optimization.

    fixed_par_indices: array_like
        Indices of fixed parameters.

    fixed_par_values: array_like
        Values of fixed parameters.

    """

    def __init__(self, objective,
                 lb, ub,
                 par_guesses=None,
                 fixed_par_indices=None,
                 fixed_par_values=None):
        self.objective = objective
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.dim = len(lb)
        self.par_guesses = par_guesses
        self.fixed_par_indices = np.asarray(fixed_par_indices)
        self.fixed_par_values = np.asarray(fixed_par_values)

    # def __init__(self, objective, model):
    #
    #     self.parameter_names = model.getParameterNames()
    #
    #     self.parameter_number = model.np()
    #
    #     self.upper_parameter_bounds = -3 * np.ones([1, self.parameter_number])
    #
    #     self.lower_parameter_bounds = 2 * np.ones([1, self.parameter_number])
    #
    #     self.starting_points = []
    #
    #     self.objective = objective
    #
    # def generate_starting_points(self, n_starts, sampling_scheme='uniform'):
    #     if sampling_scheme == 'uniform':
    #         self.starting_points = \
    #             np.random.random((n_starts, self.parameter_number)) \
    #             * (self.upper_parameter_bounds - self.lower_parameter_bounds) \
    #             + self.lower_parameter_bounds