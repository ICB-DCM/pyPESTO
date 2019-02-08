"""
Engines for parallelized multistarts
====================================

The execution of the multistarts can be parallelized in different ways, e.g.
multi-threaded or cluster-based. Note that it is not checked whether a single
multistart itself is parallelized.
"""


from .single_core import SingleCoreEngine
from .multi_process import MultiProcessEngine
from .task import Task, OptimizationTask


__all__ = [
    "SingleCoreEngine",
    "MultiProcessEngine",
]
