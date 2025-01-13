"""
Model Selection
===============

Perform model selection with a
`PEtab Select <https://petab-select.readthedocs.io/>`_ problem.
"""

from . import postprocessors
from .misc import SacessMinimizeMethod, model_to_pypesto_problem
from .model_problem import ModelProblem
from .problem import Problem

try:
    import petab_select

    del petab_select
except ImportError:
    import warnings

    warnings.warn(
        "pyPESTO's model selection methods require an installation of PEtab "
        "Select (https://github.com/PEtab-dev/petab_select). Install via "
        "`pip3 install petab-select` or `pip3 install pypesto[select]`.",
        stacklevel=1,
    )
