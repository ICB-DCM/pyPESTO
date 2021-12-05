"""
Startpoint
==========

Methods for selecting points that can be used as startpoints
for multi-start optimization.
Startpoint methods can be implemented by deriving from
:class:`pypesto.startpoint.StartpointMethod`.
"""

from .base import (
    NoStartpoints,
    StartpointMethod,
    FunctionStartpoints,
    to_startpoint_method,
)
from .uniform import (
    UniformStartpoints,
    uniform,
)
from .latin_hypercube import (
    LatinHypercubeStartpoints,
    latin_hypercube,
)
