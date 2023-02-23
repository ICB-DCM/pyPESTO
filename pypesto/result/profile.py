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
        gradnorm_path: np.ndarray = np.nan,
        exitflag_path: np.ndarray = np.nan,
        time_path: np.ndarray = np.nan,
        time_total: float = 0.0,
        n_fval: int = 0,
        n_grad: int = 0,
        n_hess: int = 0,
        message: str = None,
    ):
        super().__init__()

        # initialize profile path
        x_shape = x_path.shape
        if len(x_shape) == 1:
            self.x_path = np.zeros((x_shape[0], 1))
            self.x_path[:, 0] = x_path[:]
        else:
            self.x_path = np.zeros((x_shape[0], x_shape[1]))
            self.x_path[:, :] = x_path[:, :]

        self.fval_path = np.asarray(fval_path)
        self.ratio_path = np.asarray(ratio_path)
        self.gradnorm_path = np.asarray(gradnorm_path)
        self.exitflag_path = np.asarray(exitflag_path)
        self.time_path = np.asarray(time_path)
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
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def append_profile_point(
        self,
        x: np.ndarray,
        fval: float,
        ratio: float,
        gradnorm: float = np.nan,
        time: float = np.nan,
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

        # short function to append to numpy vectors
        def append_to_vector(field_name, val):
            field_new = np.zeros(self[field_name].size + 1)
            field_new[0:-1] = self[field_name]
            field_new[-1] = val
            self[field_name] = field_new

        # write profile path
        x_new = np.zeros((self.x_path.shape[0], self.x_path.shape[1] + 1))
        x_new[:, 0:-1] = self.x_path
        x_new[:, -1] = x
        self.x_path = x_new

        # append to other paths
        append_to_vector("fval_path", fval)
        append_to_vector("ratio_path", ratio)
        append_to_vector("gradnorm_path", gradnorm)
        append_to_vector("exitflag_path", exitflag)
        append_to_vector("time_path", time)

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


class ProfileResult:
    """
    Result of the profile() function.

    It holds a list of profile lists. Each profile list consists of a list of
    `ProfilerResult` objects, one for each parameter.
    """

    def __init__(self):
        self.list = []

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
