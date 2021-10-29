"""
Engines
=======

The execution of the multistarts can be parallelized in different ways, e.g.
multi-threaded or cluster-based. Note that it is not checked whether a single
task itself is internally parallelized.
"""

from .base import Engine
from .single_core import SingleCoreEngine
from .multi_thread import MultiThreadEngine
from .multi_process import MultiProcessEngine
from .task import Task
