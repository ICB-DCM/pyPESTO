"""
Hierarchical
============

Contains an implementation of the hierarchical optimization approach, which
decomposes the parameter estimation problem into an outer and an inner problem.
In the outer problem, only dynamic parameters are optimized.
In the inner problem, conditional on the outer solution, static parameters are
optimized.
Static parameters can be parameters affecting directly the model observables,
such as scaling factors, offsets, and noise parameters.

Hierarchical optimization has the advantage that the outer problem is typically
less complex than the full problem, and thus can be solved more efficiently.
Further, the inner problem can often be solved analytically, which is more
efficient.
Thus, hierarchical optimization can be used to speed up parameter estimation,
finding optimal values more efficiently and reliably.

The implementation in this package is based on:

* Loos et al. 2018 (https://doi.org/10.1093/bioinformatics/bty514),
  who give an analytic solution for the inner problem for scaling factors and
  noise standard deviations, for Gaussian and Laplace noise, using forward
  sensitivity analysis (FSA).
* Schmiester et al. 2020 (https://doi.org/10.1093/bioinformatics/btz581),
  who give an analytic solution for the inner problem for scaling factors,
  offsets and noise standard deviations, for Gaussian and Laplace noise,
  using adjoint sensitivity analysis (ASA). ASA allows to calculate gradients
  substantially more efficiently in high dimension.
"""

from .calculator import HierarchicalAmiciCalculator
from .parameter import InnerParameter
from .problem import InnerProblem
from .solver import AnalyticalInnerSolver, InnerSolver, NumericalInnerSolver
