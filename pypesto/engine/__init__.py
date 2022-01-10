"""
Engines
=======

The execution of the multistarts can be parallelized in different ways, e.g.
multi-threaded or cluster-based. Note that it is not checked whether a single
task itself is internally parallelized.
"""

from .base import Engine
from .multi_process import MultiProcessEngine
from .multi_thread import MultiThreadEngine
from .single_core import SingleCoreEngine
from .task import Task
