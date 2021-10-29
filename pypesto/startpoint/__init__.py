"""
Startpoint
==========

Methods for selecting points that can be used as startpoints
for multi-start optimization.
Startpoint methods can be implemented by deriving from
:class:`pypesto.startpoint.StartpointMethod`.
Handling in pypesto is then wrapped in
:func:`pypesto.startpoint.assign_startpoints`, handling e.g.
non-requirement of startpoints e.g. for global methods, and re-sampling of
non-finite points.
"""

from .base import StartpointMethod, FunctionStartpoints, to_startpoint_method
from .uniform import UniformStartpoints, uniform
from .latin_hypercube import LatinHypercubeStartpoints, latin_hypercube
from .assign import assign_startpoints
