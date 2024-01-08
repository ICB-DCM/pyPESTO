"""
Semi-quantitative data integration
==================================

Contains the implementation of a spline approximation approach, applied for
integration of semi-quantitative data in ODE modeling, where the data is
assumed to have an unknown monotone relationship with the model output. This
relationship is approximated by a piecewise linear (spline) function, which
is numerically optimized to fit the data. This constitutes the inner subproblem
of the hierarchical optimization problem.

An example of parameter estimation with semi-quantitative data
can be found in pypesto/doc/examples/semiquantitative_data.ipynb.
"""

from .calculator import SemiquantCalculator
from .parameter import SplineInnerParameter
from .problem import SemiquantProblem
from .solver import SemiquantInnerSolver
