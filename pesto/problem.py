import numpy as np

class Problem:

    def __init__(self, objective, model):

        self.parameter_names = model.getParameterNames()

        self.parameter_number = model.np()

        self.upper_parameter_bounds = -3 * np.ones([1, self.parameter_number])

        self.lower_parameter_bounds = 2 * np.ones([1, self.parameter_number])

        self.starting_points = []

        self.objective = objective

    def generate_starting_points(self, n_starts, sampling_scheme='uniform'):
        if sampling_scheme == 'uniform':
            self.starting_points = \
                np.random.random((n_starts, self.parameter_number)) \
                * (self.upper_parameter_bounds - self.lower_parameter_bounds) \
                + self.lower_parameter_bounds