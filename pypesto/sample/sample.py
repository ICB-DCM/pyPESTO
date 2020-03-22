import logging

from ..problem import Problem
from ..result import Result
from .result import SamplerResult
from .sampler import Sampler

logger = logging.getLogger(__name__)


def parameter_sample(
        problem: Problem,
        sampler: Sampler,
        result: Result = None
) -> Result:
    """
    This is the main function to call to do parameter sampling.

    Parameters
    ----------
    problem:
        The problem to be solved.
    sampler:
        The sampler to perform the actual sampling.
    result:
        A result to write to.

    Returns
    -------
    result:
        A result with filled in sample_options part.
    """
    if result is None:
        result = Result(problem)

    obj = problem.objective
    sample_result: SamplerResult = sampler.sample(obj)

    result.sample_result = sample_result

    return result
