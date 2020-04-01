import logging

from ..problem import Problem
from ..result import Result
from .sampler import Sampler
from .pymc3 import Pymc3Sampler

logger = logging.getLogger(__name__)


def sample(
        problem: Problem,
        sampler: Sampler = None,
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

    if sampler is None:
        sampler = Pymc3Sampler()

    sample_result = sampler.sample(problem=problem)

    result.sample_result = sample_result

    return result
