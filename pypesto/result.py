"""
Result
======

The pypesto.Result object contains all results generated by
the pypesto components. It contains sub-results for
optimization, profiling, sampling.

"""

import pandas as pd
import copy
from typing import Sequence


class OptimizeResult:
    """
    Result of the minimize() function.
    """

    def __init__(self):
        self.list = []

    def append(
            self,
            optimizer_result: 'optimize.OptimizerResult',  # noqa: F821
    ):
        """
        Append an optimizer result to the result object.

        Parameters
        ----------
        optimizer_result:
            The result of one (local) optimizer run.
        """

        self.list.append(optimizer_result)

    def sort(self):
        """
        Sort the optimizer results by function value fval (ascending).
        """

        self.list = sorted(self.list, key=lambda res: res.fval)

    def as_dataframe(self, keys=None) -> pd.DataFrame:
        """
        Get as pandas DataFrame. If keys is a list,
        return only the specified values.
        """

        lst = self.as_list(keys)

        df = pd.DataFrame(lst)

        return df

    def as_list(self, keys=None) -> Sequence:
        """
        Get as list. If keys is a list,
        return only the specified values.

        Parameters
        ----------
        keys: list(str), optional
            Labels of the field to extract.
        """

        lst = self.list

        if keys is not None:
            lst = [{key: res[key] for key in keys} for res in lst]

        return lst

    def get_for_key(self, key) -> list:
        """
        Extract the list of values for the specified key as a list.
        """

        return [res[key] for res in self.list]


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
            profiler_result: 'profile.ProfilerResult' = None,  # noqa: F821
            profile_list: int = None) -> None:
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
            profiler_result: 'profile.ProfilerResult',  # noqa: F821
            i_par: int,
            profile_list: int = None) -> None:
        """Write a profiler result to the result object at `i_par` of profile
        list `profile_list`.

        Parameters
        ----------
        profiler_result:
            The result of one (local) profiler run.
        i_par:
            Integer specifying the parameter index.
        profile_list:
            Index specifying the profile list. Defaults to the last list.
        """
        if profile_list is None:
            profile_list = -1  # last
        self.list[profile_list][i_par] = copy.deepcopy(profiler_result)

    def get_profiler_result(
            self, i_par: int, profile_list: int = None
    ):
        """
        Get theprofiler result at parameter index `i_par` of profile list
        `profile_list`.

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


class SampleResult:
    """
    Result of the sample() function.
    """

    def __init__(self):
        pass


class Result:
    """
    Universal result object for pypesto.
    The algorithms like optimize, profile,
    sample fill different parts of it.

    Attributes
    ----------

    problem: pypesto.Problem
        The problem underlying the results.

    optimize_result:
        The results of the optimizer runs.

    profile_result:
        The results of the profiler run.

    sample_result:
        The results of the sampler run.

    """

    def __init__(self, problem=None):
        self.problem = problem
        self.optimize_result = OptimizeResult()
        self.profile_result = ProfileResult()
        self.sample_result = SampleResult()
