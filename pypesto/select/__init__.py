"""
Model Selection
===============
TODO make import optional
    - remove from pypesto/__init__.py
    - import with `import pypesto.select as ms`?
"""

from .criteria import (
    calculate_aic,
    calculate_aicc,
    calculate_bic,
)
from .method import ModelSelectorMethod
from .method_stepwise import ForwardSelector
from .misc import (
    row2problem,
    unpack_file,
)
from .problem import ModelSelectionProblem
from .selector import ModelSelector
