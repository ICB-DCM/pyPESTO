"""Overall result."""

from .optimize import OptimizeResult
from .profile import ProfileResult
from .sample import SampleResult


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

    def __init__(
        self,
        problem=None,
        optimize_result: OptimizeResult = None,
        profile_result: ProfileResult = None,
        sample_result: SampleResult = None,
    ):
        self.problem = problem
        self.optimize_result = optimize_result or OptimizeResult()
        self.profile_result = profile_result or ProfileResult()
        self.sample_result = sample_result or SampleResult()

    def summary(self, full: bool = False, show_hess: bool = True) -> str:
        """
        Get summary of the object.

        Parameters
        ----------
        full:
            If True, print full vectors including fixed parameters.
        show_hess:
            If True, display the Hessian of the OptimizeResult.
        """
        return self.optimize_result.summary(full=full, show_hess=show_hess)
