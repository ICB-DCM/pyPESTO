"""
Startpoint
==========

Methods for selecting points that can be used as startpoints
for multi-start optimization.
Startpoint methods can be implemented by deriving from
:class:`pypesto.startpoint.StartpointMethod`.
"""

from .base import (
    CheckedStartpoints,
    FunctionStartpoints,
    NoStartpoints,
    StartpointMethod,
    to_startpoint_method,
)
from .latin_hypercube import LatinHypercubeStartpoints, latin_hypercube
from .uniform import UniformStartpoints, uniform
