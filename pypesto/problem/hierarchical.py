import logging
from typing import Iterable, List, Optional, SupportsFloat, SupportsInt, Union

import numpy as np

from .base import Problem

SupportsFloatIterableOrValue = Union[Iterable[SupportsFloat], SupportsFloat]
SupportsIntIterableOrValue = Union[Iterable[SupportsInt], SupportsInt]

logger = logging.getLogger(__name__)


class HierarchicalProblem(Problem):
    """
    The Hierarchical Problem.

    A hierarchical problem is a problem with a nested structure: One or
    multiple inner problems are nested inside the outer problem. The inner
    problems are optimized for each evaluation of the outer problem. The
    objective's calculator is used to collect the inner problems' objective
    values.

    Parameters
    ----------
    hierarchical:
        A flag indicating the problem is hierarchical.
    inner_x_names:
        Names of the inner optimization parameters. Only relevant if
        hierarchical is True. Contains the names of easily interpretable
        inner parameters only, e.g. noise parameters, scaling factors, offsets.
    inner_lb, inner_ub:
        The lower and upper bounds for the inner optimization parameters.
        Only relevant if hierarchical is True. Contains the bounds of easily
        interpretable inner parameters only, e.g. noise parameters, scaling
        factors, offsets.
    """

    def __init__(
        self,
        inner_x_names: Optional[Iterable[str]] = None,
        inner_lb: Optional[Union[np.ndarray, List[float]]] = None,
        inner_ub: Optional[Union[np.ndarray, List[float]]] = None,
        **problem_kwargs: dict,
    ):
        super().__init__(**problem_kwargs)

        if inner_x_names is None:
            inner_x_names = (
                self.objective.calculator.get_interpretable_inner_par_ids()
            )
        if len(set(inner_x_names)) != len(inner_x_names):
            raise ValueError("Parameter names inner_x_names must be unique")
        self.inner_x_names = inner_x_names

        if inner_lb is None or inner_ub is None:
            (
                default_inner_lb,
                default_inner_ub,
            ) = self.objective.calculator.get_interpretable_inner_par_bounds()
            inner_lb = default_inner_lb if inner_lb is None else inner_lb
            inner_ub = default_inner_ub if inner_ub is None else inner_ub

        if len(inner_lb) != len(inner_ub):
            raise ValueError("Parameter bounds must have same length")
        if len(inner_lb) != len(inner_x_names):
            raise ValueError(
                "Parameter bounds must have same length as parameter names"
            )

        self.inner_lb = np.array(inner_lb)
        self.inner_ub = np.array(inner_ub)
