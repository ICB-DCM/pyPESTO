from typing import Callable

from petab_select.constants import TYPE_PATH


TYPE_POSTPROCESSOR = Callable[['ModelSelectionProblem'], None]
