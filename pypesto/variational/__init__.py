"""
Variational inference
======

Find the best variational approximation in a given family to a distribution from which we can sample.
"""

from .variational_inference import (
    eval_variational_log_density,
    variational_fit,
)
