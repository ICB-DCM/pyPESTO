"""
Relative data integration
=========================

Contains an implementation of the hierarchical inner subproblem for relative data.
In this inner problem, the scaling factors, offsets, and noise standard deviations
are optimized, conditional on the outer dynamical parameters. The inner problem
can be solved analytically.

An example of parameter estimation with relative data
can be found in pypesto/doc/examples/relative_data.ipynb.

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

from .calculator import RelativeAmiciCalculator
from .problem import RelativeInnerProblem
from .solver import AnalyticalInnerSolver, NumericalInnerSolver
