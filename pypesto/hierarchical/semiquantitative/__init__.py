"""
Spline approximation
====================

Contains the implementation of a spline approximation approach, applied for integration
of semi-quantitative data in ODE modeling, where the data is assumed to have
an unknown monotone relationship with the model output. This relationship is
approximated by a piecewise linear (spline) function, which is numerically
optimized to fit the data. This constitutes the inner subproblem of the
hierarchical optimization problem.
"""

from .calculator import SemiquantitativeCalculator
from .parameter import SemiquantitativeInnerParameter
from .problem import SemiquantitativeInnerProblem
from .solver import SemiquantitativeInnerSolver
