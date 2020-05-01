"""
This is for testing the pypesto.model_selection.
"""
import tempfile
import numpy as np
from pypesto.model_selection import (ModelSelectionProblem, ModelSelection)

from pypesto.model_selection import (
    ModelSelectionHelper,
    ModelSelector,
    ForwardSelector,
    ModelSelectionProblem
)

def test_model_selection():
    # model_selection.tsv
    petab_problem = None
    model_selection = ModelSelection(petab_problem, 'model_selection.tsv')
    # sm = model_selection.get_smallest_order_problem()
    res = model_selection.forward_selection()

def test_select_model()
    petab_problem = None
    model_specification_file = 'model_selection.tsv'
    selector = ModelSelector(petab_problem, model_specification_file)
    result = selector.select('forward', 'AIC')
