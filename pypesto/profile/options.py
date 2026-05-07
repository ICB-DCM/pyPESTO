from typing import Union


class ProfileOptions(dict):
    """
    Options for optimization based profiling.

    Attributes
    ----------
    default_step_size:
        Default step size of the profiling routine along the profile path
        (adaptive step lengths algorithms will only use this as a first guess
        and then refine the update).
    default_step_size_relative:
        Relative default step size for wide `lin`-scale parameters, expressed
        as a fraction of the full parameter span `ub - lb`. The effective
        default step size is the maximum of the absolute and relative values.
    min_step_size:
        Lower bound for the step size in adaptive methods.
    min_step_size_relative:
        Relative minimum step size for wide `lin`-scale parameters, expressed
        as a fraction of the full parameter span `ub - lb`. The effective
        minimum step size is the maximum of the absolute and relative values.
    max_step_size:
        Upper bound for the step size in adaptive methods.
    max_step_size_relative:
        Relative maximum step size for wide `lin`-scale parameters, expressed
        as a fraction of the full parameter span `ub - lb`. The effective
        maximum step size is the maximum of the absolute and relative values.
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
    adaptive_target_scaling_factor:
        The scaling factor of the next_obj_target in next guess generation.
        Larger values result in larger next_guess step size (must be > 1).
    whole_path:
        Whether to profile the whole bounds or only till we get below the
        ratio.
    profile_n_starts:
        Number of optimization starts attempted at each profile point.
    profile_sampling_sigma:
        Standard deviation used for each free coordinate when sampling
        additional optimization startpoints from a Gaussian centered at the
        point suggested by `profile_next_guess`, in the reduced parameter
        space. The profiled parameter itself is already fixed and is therefore
        not perturbed.
    step_size_precheck_mode:
        Behavior of the profile step-size precheck.
        Use ``"off"`` to disable the precheck, ``"warn"`` to emit a warning
        for suspiciously small steps, and ``"raise"`` to raise an error for
        extreme cases.
    """

    def __init__(
        self,
        default_step_size: float = 0.02,
        default_step_size_relative: float = 0.01,
        min_step_size: float = 0.01,
        min_step_size_relative: float = 0.005,
        max_step_size: float = 0.2,
        max_step_size_relative: float = 0.04,
        step_size_factor: float = 1.25,
        delta_ratio_max: float = 0.1,
        ratio_min: float = 0.145,
        reg_points: int = 10,
        reg_order: int = 4,
        adaptive_target_scaling_factor: float = 1.5,
        whole_path: bool = False,
        profile_n_starts: int = 6,
        profile_sampling_sigma: float = 0.01,
        step_size_precheck_mode: str = "warn",
    ):
        super().__init__()

        self.default_step_size = default_step_size
        self.default_step_size_relative = default_step_size_relative
        self.min_step_size = min_step_size
        self.min_step_size_relative = min_step_size_relative
        self.max_step_size = max_step_size
        self.max_step_size_relative = max_step_size_relative
        self.ratio_min = ratio_min
        self.step_size_factor = step_size_factor
        self.delta_ratio_max = delta_ratio_max
        self.reg_points = reg_points
        self.reg_order = reg_order
        self.adaptive_target_scaling_factor = adaptive_target_scaling_factor
        self.whole_path = whole_path
        self.profile_n_starts = profile_n_starts
        self.profile_sampling_sigma = profile_sampling_sigma
        self.step_size_precheck_mode = step_size_precheck_mode

        self.validate()

    def __getattr__(self, key):
        """Allow usage of keys like attributes."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def create_instance(
        maybe_options: Union["ProfileOptions", dict],
    ) -> "ProfileOptions":
        """
        Return a valid options object.

        Parameters
        ----------
        maybe_options: ProfileOptions or dict
        """
        if isinstance(maybe_options, ProfileOptions):
            return maybe_options
        options = ProfileOptions(**maybe_options)
        return options

    def validate(self):
        """Check if options are valid.

        Raises ``ValueError`` if current settings aren't valid.
        """
        if self.min_step_size <= 0:
            raise ValueError("min_step_size must be > 0.")
        if self.max_step_size <= 0:
            raise ValueError("max_step_size must be > 0.")
        if self.min_step_size > self.max_step_size:
            raise ValueError("min_step_size must be <= max_step_size.")
        if self.default_step_size <= 0:
            raise ValueError("default_step_size must be > 0.")
        if self.default_step_size > self.max_step_size:
            raise ValueError("default_step_size must be <= max_step_size.")
        if self.default_step_size < self.min_step_size:
            raise ValueError("default_step_size must be >= min_step_size.")
        if self.default_step_size_relative <= 0:
            raise ValueError("default_step_size_relative must be > 0.")
        if self.min_step_size_relative <= 0:
            raise ValueError("min_step_size_relative must be > 0.")
        if self.max_step_size_relative <= 0:
            raise ValueError("max_step_size_relative must be > 0.")
        if self.min_step_size_relative > self.default_step_size_relative:
            raise ValueError(
                "min_step_size_relative must be <= default_step_size_relative."
            )
        if self.default_step_size_relative > self.max_step_size_relative:
            raise ValueError(
                "default_step_size_relative must be <= max_step_size_relative."
            )

        if self.adaptive_target_scaling_factor < 1:
            raise ValueError("adaptive_target_scaling_factor must be > 1.")
        if self.profile_n_starts < 1:
            raise ValueError("profile_n_starts must be >= 1.")
        if self.profile_sampling_sigma <= 0:
            raise ValueError("profile_sampling_sigma must be > 0.")
        if self.step_size_precheck_mode not in {"off", "warn", "raise"}:
            raise ValueError(
                "step_size_precheck_mode must be one of "
                "{'off', 'warn', 'raise'}."
            )
