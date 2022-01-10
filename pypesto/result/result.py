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

    def __init__(self, problem=None):
        self.problem = problem
        self.optimize_result = OptimizeResult()
        self.profile_result = ProfileResult()
        self.sample_result = SampleResult()
