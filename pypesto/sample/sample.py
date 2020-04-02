import logging
import numpy as np

from ..problem import Problem
from ..result import Result
from .sampler import Sampler
from .pymc3 import Pymc3Sampler

logger = logging.getLogger(__name__)


def sample(
        problem: Problem,
        sampler: Sampler = None,
        x0: np.ndarray = None,
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

    if x0 is None:
        result.optimize_result.sort()
        xs = result.optimize_result.get_for_key('x')
        if len(xs) > 0:
            x0 = xs[0]

    if sampler is None:
        sampler = Pymc3Sampler()

    sample_result = sampler.sample(problem=problem, x0=x0)

    result.sample_result = sample_result

    return result
