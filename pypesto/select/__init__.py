"""
Model Selection
===============

Perform model selection with a PEtab Select problem.
"""

from .misc import model_to_pypesto_problem
from .problem import Problem

try:
    import petab_select
    del petab_select
except ImportError:
    import warnings
    warnings.warn(
        "pyPESTO's model selection methods require an installation of PEtab "
        "Select (https://github.com/PEtab-dev/petab_select). Install via "
        "`pip3 install petab-select` or `pip3 install pypesto[select]`."
    )
