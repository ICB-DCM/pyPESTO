"""
Variational inference
======

Find the best variational approximation in a given family to a distribution from which we can sample.
"""

from .pymc import PymcVariational
from .variational_inference import variational_fit
