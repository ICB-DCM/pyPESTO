import warnings
from typing import Union

#: Deprecated ``ProfileOptions`` step-size names mapped to their replacements.
#: Kept for backward compatibility, both as constructor arguments and as
#: attributes.
_DEPRECATED_STEP_SIZE_NAMES = {
    "default_step_size": "default_step_size_absolute",
    "min_step_size": "min_step_size_absolute",
    "max_step_size": "max_step_size_absolute",
}


class ProfileOptions(dict):
    """
    Options for optimization based profiling.

    Step sizes can be configured as absolute values or relative fractions of
    the parameter span. A family is disabled by setting its default step size
    to `0`. For each profiled parameter, pyPESTO uses either the full absolute
    family or the full relative family, whichever has the larger default step
    size, i.e. whichever of `default_step_size_absolute` and
    `default_step_size_relative * (ub - lb)` is larger.

    Attributes
    ----------
    default_step_size_absolute:
        Default absolute profile step size. Set to `0` to disable.
    default_step_size_relative:
        Default relative profile step size, as fraction of `ub - lb`. Set to
        `0` to disable.
    min_step_size_absolute:
        Minimum absolute step size in adaptive methods.
    min_step_size_relative:
        Minimum relative step size, as fraction of `ub - lb`.
    max_step_size_absolute:
        Maximum absolute step size in adaptive methods.
    max_step_size_relative:
        Maximum relative step size, as fraction of `ub - lb`.
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
        Controls the step-size precheck, which estimates how many profile
        steps the resolved step sizes imply and reports suspiciously small
        steps. One of ``"off"`` (disable the precheck), ``"warn"`` (only ever
        emit a warning), or ``"raise"`` (raise an error only for extreme,
        worst-case estimates, and warn otherwise).
    """

    def __init__(
        self,
        default_step_size_absolute: float = 0.02,
        default_step_size_relative: float = 0.0025,
        min_step_size_absolute: float = 0.01,
        min_step_size_relative: float = 0.00125,
        max_step_size_absolute: float = 0.2,
        max_step_size_relative: float = 0.025,
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
        default_step_size: float | None = None,
        min_step_size: float | None = None,
        max_step_size: float | None = None,
    ):
        super().__init__()

        # Backward compatibility: the absolute step-size arguments were
        # renamed. If an old name is passed, it overrides the new one.
        if default_step_size is not None:
            warnings.warn(
                "`default_step_size` is deprecated. Use "
                "`default_step_size_absolute` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            default_step_size_absolute = default_step_size
        if min_step_size is not None:
            warnings.warn(
                "`min_step_size` is deprecated. Use "
                "`min_step_size_absolute` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            min_step_size_absolute = min_step_size
        if max_step_size is not None:
            warnings.warn(
                "`max_step_size` is deprecated. Use "
                "`max_step_size_absolute` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            max_step_size_absolute = max_step_size

        self.default_step_size_absolute = default_step_size_absolute
        self.default_step_size_relative = default_step_size_relative
        self.min_step_size_absolute = min_step_size_absolute
        self.min_step_size_relative = min_step_size_relative
        self.max_step_size_absolute = max_step_size_absolute
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
        if key in _DEPRECATED_STEP_SIZE_NAMES:
            new_key = _DEPRECATED_STEP_SIZE_NAMES[key]
            warnings.warn(
                f"`{key}` is deprecated. Use `{new_key}` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self[new_key]
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

        def validate_step_size_family(family: str) -> bool:
            default_step_size = self[f"default_step_size_{family}"]
            min_step_size = self[f"min_step_size_{family}"]
            max_step_size = self[f"max_step_size_{family}"]

            if default_step_size < 0:
                raise ValueError(f"default_step_size_{family} must be >= 0.")
            if default_step_size == 0:
                return False
            if min_step_size <= 0:
                raise ValueError(f"min_step_size_{family} must be > 0.")
            if max_step_size <= 0:
                raise ValueError(f"max_step_size_{family} must be > 0.")
            if min_step_size > default_step_size:
                raise ValueError(
                    f"min_step_size_{family} must be <= "
                    f"default_step_size_{family}."
                )
            if default_step_size > max_step_size:
                raise ValueError(
                    f"default_step_size_{family} must be <= "
                    f"max_step_size_{family}."
                )
            return True

        absolute_enabled = validate_step_size_family("absolute")
        relative_enabled = validate_step_size_family("relative")
        if not absolute_enabled and not relative_enabled:
            raise ValueError(
                "At least one step-size family must be enabled by setting "
                "default_step_size_absolute > 0 or "
                "default_step_size_relative > 0."
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
