"""
Ordinal and censored data integration
=====================================

Contains the implementation of the optimal scaling approach, applied for integration
of ordinal and censored data in ODE modeling. Each ordinal datapoint is defined as
being part of a category, where the mutual ordering of the categories of the same observable
is known. The category interval bounds are numerically optimized and quantitative surrogate
measurements are calculated to represent the ordinal measurements. This constitutes
the inner subproblem of the hierarchical optimization problem. For censored data, as the
category interval bounds are known, the surrogate measurements are directly calculated.

An example of parameter estimation with ordinal data
can be found in pypesto/doc/examples/ordinal_data.ipynb.
An example of parameter estimation with censored data
can be found in pypesto/doc/examples/censored_data.ipynb.

Details on the optimal scaling approach can be found in Shepard, 1962 (https://doi.org/10.1007/BF02289621).
Details on the application of the gradient-based optimal scaling approach to mechanistic modeling
with ordinal data can be found in Schmiester et al. 2020 (https://doi.org/10.1007/s00285-020-01522-w)
and Schmiester et al. 2021 (https://doi.org/10.1093/bioinformatics/btab512).
"""

from .calculator import OrdinalCalculator
from .parameter import OrdinalParameter
from .problem import OrdinalProblem
from .solver import OrdinalInnerSolver
