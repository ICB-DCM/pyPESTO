"""
Optimal scaling
===============

Contains the implementation of the optimal scaling approach, applied for integration
of ordinal data in ODE modeling. Each ordinal datapoint is defined as being part of
a category, where the mutual ordering of the categories of the same group is known.
The category interval bounds are numerically optimized and quantitative surrogate
measurements are calculated to represent the ordinal measurements. This constitutes
the inner subproblem of the hierarchical optimization problem.

Details on the optimal scaling approach can be found in Shepard, 1962, https://doi.org/10.1007/BF02289621.
Details on the application of the gradient-based optimal scaling approach to mechanistic modeling
with ordinal data can be found in Schmiester et al. 2020 (https://doi.org/10.1007/s00285-020-01522-w)
and Schmiester et al. 2021 (https://doi.org/10.1093/bioinformatics/btab512).
"""

from .calculator import OptimalScalingAmiciCalculator
from .problem import OptimalScalingProblem
from .solver import OptimalScalingInnerSolver
