"""
Model Selection
===============
TODO make import optional
    - remove from pypesto/__init__.py
    - import with `import pypesto.model_selection as ms`?
"""

from .method_stepwise import ForwardSelector
from .method import ModelSelectorMethod
from .misc import row2problem, unpack_file
from .selector import ModelSelector
from .problem import ModelSelectionProblem
