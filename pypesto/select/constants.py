from typing import Callable

from petab_select.constants import TYPE_PATH  # noqa: F401


TYPE_POSTPROCESSOR = Callable[["ModelSelectionProblem"], None]  # noqa: F821
