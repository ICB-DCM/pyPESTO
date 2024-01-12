"""
Hierarchical
============

Contains an implementation of the hierarchical optimization approach, which
decomposes the parameter estimation problem into an outer and an inner problem.
In the outer problem, only dynamic parameters are optimized.
In the inner problem, conditional on the outer solution, static parameters are
optimized.
Static parameters can be parameters affecting only the model observables,
such as scaling factors, offsets, and noise parameters. Further, they can be
spline parameters, which are used to approximate non-linear measurement mappings
of semi-quantitative data. Finally, they can be optimal scaling parameters,
which are used for the integration of ordinal data.

Hierarchical optimization has the advantage that the outer problem is typically
less complex than the full problem, and thus can be solved more efficiently.
Further, in the relative data case, the inner problem can be solved analytically,
which is more efficient.
Thus, hierarchical optimization can be used to speed up parameter estimation,
finding optimal values more efficiently and reliably.
"""

from .base_parameter import InnerParameter
from .base_problem import InnerProblem
from .base_solver import InnerSolver
from .inner_calculator_collector import InnerCalculatorCollector
