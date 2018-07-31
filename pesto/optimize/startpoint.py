def uniform(lb, ub, n_starts):
    """"""

    self.starting_points = \
                 np.random.random((n_starts, self.parameter_number)) \
               * (self.upper_parameter_bounds - self.lower_parameter_bounds) \
               + self.lower_parameter_bounds


def latin_hypercube(lb, ub, n_starts):
    pass
