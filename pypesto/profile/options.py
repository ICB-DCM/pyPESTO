from typing import Dict, Union


class ProfileOptions(dict):
    """
    Options for optimization based profiling.

    Parameters
    ----------
    default_step_size:
        Default step size of the profiling routine along the profile path
        (adaptive step lengths algorithms will only use this as a first guess
        and then refine the update).
    min_step_size:
        Lower bound for the step size in adaptive methods.
    max_step_size:
        Upper bound for the step size in adaptive methods.
    step_size_factor:
        Adaptive methods recompute the likelihood at the predicted point and
        try to find a good step length by a sort of line search algorithm.
        This factor controls step handling in this line search.
    delta_ratio_max:
        Maximum allowed drop of the posterior ratio between two profile steps.
    ratio_min:
        Lower bound for likelihood ratio of the profile, based on inverse
        chi2-distribution.
        The default 0.145 is slightly lower than the 95% quantile 0.1465 of a
        chi2 distribution with one degree of freedom.
    reg_points:
        Number of profile points used for regression in regression based
        adaptive profile points proposal.
    reg_order:
        Maximum degree of regression polynomial used in regression based
        adaptive profile points proposal.
    magic_factor_obj_value:
        There is this magic factor in the old profiling code which slows down
        profiling at small ratios (must be >= 0 and < 1).
    """

    def __init__(self,
                 default_step_size: float = 0.01,
                 min_step_size: float = 0.001,
                 max_step_size: float = 1.,
                 step_size_factor: float = 1.25,
                 delta_ratio_max: float = 0.1,
                 ratio_min: float = 0.145,
                 reg_points: int = 10,
                 reg_order: int = 4,
                 magic_factor_obj_value: float = 0.5):
        super().__init__()

        self.default_step_size = default_step_size
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.ratio_min = ratio_min
        self.step_size_factor = step_size_factor
        self.delta_ratio_max = delta_ratio_max
        self.reg_points = reg_points
        self.reg_order = reg_order
        self.magic_factor_obj_value = magic_factor_obj_value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def create_instance(
            maybe_options: Union['ProfileOptions', Dict]
    ) -> 'ProfileOptions':
        """
        Returns a valid options object.

        Parameters
        ----------
        maybe_options: ProfileOptions or dict
        """
        if isinstance(maybe_options, ProfileOptions):
            return maybe_options
        options = ProfileOptions(**maybe_options)
        return options
