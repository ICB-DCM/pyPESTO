"""Profiling result."""

import copy

import numpy as np


class ProfilerResult(dict):
    """
    The result of a profiler run.

    The standardized return value from pypesto.profile, which can
    either be initialized from an OptimizerResult or from an existing
    ProfilerResult (in order to extend the computation).

    Can be used like a dict.

    Attributes
    ----------
    x_path:
        The path of the best found parameters along the profile
        (Dimension: n_par x n_profile_points)
    fval_path:
        The function values, fun(x), along the profile.
    ratio_path:
        The ratio of the posterior function along the profile.
    gradnorm_path:
        The gradient norm along the profile.
    exitflag_path:
        The exitflags of the optimizer along the profile.
    time_path:
        The computation time of the optimizer runs along the profile.
    time_total:
        The total computation time for the profile.
    n_fval:
        Number of function evaluations.
    n_grad:
        Number of gradient evaluations.
    n_hess:
        Number of Hessian evaluations.
    color_path:
        The color of the profile path. Signifies types of steps made.
        Red indicates a step for which min_step_size was reduced, blue
        indicates a step for which max_step_size was increased, and green
        indicates a step for which the profiler had to resample the parameter
        vector due to optimization failure of the previous two. Black
        indicates a step for which none of the above was necessary.
    message:
        Textual comment on the profile result.

    Notes
    -----
    Any field not supported by the profiler or the profiling optimizer is
    filled with None. Some fields are filled by pypesto itself.
    """

    def __init__(
        self,
        x_path: np.ndarray,
        fval_path: np.ndarray,
        ratio_path: np.ndarray,
        gradnorm_path: np.ndarray = None,
        exitflag_path: np.ndarray = None,
        time_path: np.ndarray = None,
        color_path: np.ndarray = None,
        time_total: float = 0.0,
        n_fval: int = 0,
        n_grad: int = 0,
        n_hess: int = 0,
        message: str = None,
    ):
        super().__init__()

        # initialize profile path
        if not x_path.ndim == 2:
            raise ValueError("x_path must be a 2D array.")

        self.x_path = x_path.copy()
        self.fval_path = fval_path.copy()
        self.ratio_path = ratio_path.copy()

        if gradnorm_path is None:
            self.gradnorm_path = np.full(x_path.shape[1], np.nan)
        else:
            self.gradnorm_path = gradnorm_path.copy()

        if exitflag_path is None:
            self.exitflag_path = np.full(x_path.shape[1], np.nan)
        else:
            self.exitflag_path = exitflag_path.copy()

        if time_path is None:
            self.time_path = np.full(x_path.shape[1], np.nan)
        else:
            self.time_path = time_path.copy()

        if color_path is None:
            self.color_path = np.full(
                (x_path.shape[1], 4), np.array([1, 0, 0, 0.3])
            )
        else:
            self.color_path = color_path.copy()

        if (
            not self.x_path.shape[1]
            == len(self.fval_path)
            == len(self.ratio_path)
            == len(self.gradnorm_path)
            == len(self.exitflag_path)
            == len(self.time_path)
        ):
            raise ValueError(
                "x_path, fval_path, ratio_path, gradnorm_path, exitflag_path, "
                "time_path must have the same length."
            )

        self.time_total = time_total
        self.n_fval = n_fval
        self.n_grad = n_grad
        self.n_hess = n_hess
        self.message = message

    def __getattr__(self, key):
        """Allow usage of keys like attributes."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def append_profile_point(
        self,
        x: np.ndarray,
        fval: float,
        ratio: float,
        gradnorm: float = np.nan,
        time: float = np.nan,
        color: np.ndarray = np.nan,
        exitflag: float = np.nan,
        n_fval: int = 0,
        n_grad: int = 0,
        n_hess: int = 0,
    ) -> None:
        """
        Append a new point to the profile path.

        Parameters
        ----------
        x:
            The parameter values.
        fval:
            The function value at `x`.
        ratio:
            The ratio of the function value at `x` by the optimal function
            value.
        gradnorm:
            The gradient norm at `x`.
        time:
            The computation time to find `x`.
        color:
            The color of the profile path. Signifies types of steps made.
        exitflag:
            The exitflag of the optimizer (useful if an optimization was
            performed to find `x`).
        n_fval:
            Number of function evaluations performed to find `x`.
        n_grad:
            Number of gradient evaluations performed to find `x`.
        n_hess:
            Number of Hessian evaluations performed to find `x`.
        """
        self.x_path = np.hstack((self.x_path, x[..., np.newaxis]))
        self.fval_path = np.hstack((self.fval_path, fval))
        self.ratio_path = np.hstack((self.ratio_path, ratio))
        self.gradnorm_path = np.hstack((self.gradnorm_path, gradnorm))
        self.exitflag_path = np.hstack((self.exitflag_path, exitflag))
        self.time_path = np.hstack((self.time_path, time))
        self.color_path = np.vstack((self.color_path, color))

        # increment the time and f_eval counters
        self.time_total += time
        self.n_fval += n_fval
        self.n_grad += n_grad
        self.n_hess += n_hess

    def flip_profile(self) -> None:
        """
        Flip the profiling direction (left-right).

        Profiling direction needs to be changed once (if the profile is new),
        or twice if we append to an existing profile. All profiling paths
        are flipped in-place.
        """
        self.x_path = np.fliplr(self.x_path)
        self.fval_path = np.flip(self.fval_path)
        self.ratio_path = np.flip(self.ratio_path)
        self.gradnorm_path = np.flip(self.gradnorm_path)
        self.exitflag_path = np.flip(self.exitflag_path)
        self.time_path = np.flip(self.time_path)
        self.color_path = np.flip(self.color_path, axis=0)


class ProfileResult:
    """
    Result of the profile() function.

    It holds a list of profile lists. Each profile list consists of a list of
    `ProfilerResult` objects, one for each parameter.
    """

    def __init__(self):
        self.list: list[list[ProfilerResult]] = []

    def append_empty_profile_list(self) -> int:
        """Append an empty profile list to the list of profile lists.

        Returns
        -------
        index:
            The index of the created profile list.
        """
        self.list.append([])
        return len(self.list) - 1

    def append_profiler_result(
        self,
        profiler_result: ProfilerResult = None,
        profile_list: int = None,
    ) -> None:
        """Append the profiler result to the profile list.

        Parameters
        ----------
        profiler_result:
            The result of one profiler run for a parameter, or None if to be
            left empty.
        profile_list:
            Index specifying the profile list to which we want to append.
            Defaults to the last list.
        """
        if profile_list is None:
            profile_list = -1  # last
        profiler_result = copy.deepcopy(profiler_result)
        self.list[profile_list].append(profiler_result)

    def set_profiler_result(
        self,
        profiler_result: ProfilerResult,
        i_par: int,
        profile_list: int = None,
    ) -> None:
        """
        Write a profiler result to the result object.

        Parameters
        ----------
        profiler_result:
            The result of one (local) profiler run.
        i_par:
            Integer specifying the parameter index where to put
            profiler_result.
        profile_list:
            Index specifying the profile list. Defaults to the last list.
        """
        if profile_list is None:
            profile_list = -1  # last
        self.list[profile_list][i_par] = copy.deepcopy(profiler_result)

    def get_profiler_result(self, i_par: int, profile_list: int = None):
        """
        Get the profiler result at parameter index `i_par` of `profile_list`.

        Parameters
        ----------
        i_par:
            Integer specifying the profile index.
        profile_list:
            Index specifying the profile list. Defaults to the last list.
        """
        if profile_list is None:
            profile_list = -1  # last
        return self.list[profile_list][i_par]
