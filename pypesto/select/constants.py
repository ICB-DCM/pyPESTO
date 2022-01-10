from typing import Callable

from petab_select.constants import TYPE_PATH  # noqa: F401


TYPE_POSTPROCESSOR = Callable[["ModelProblem"], None]  # noqa: F821
