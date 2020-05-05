"""
This is for testing the pypesto.model_selection.
"""
import tempfile
import numpy as np
from pypesto.model_selection import (
    ModelSelector,
    ModelSelectionProblem,
    unpack_file
)

# def test_model_selection():
#    # model_selection.tsv
#    petab_problem = None
#    model_selection = ModelSelection(petab_problem, 'model_selection.tsv')
#    # sm = model_selection.get_smallest_order_problem()
#    res = model_selection.forward_selection()

# def test_select_model():
#    petab_problem = None
#    model_specification_file = 'model_selection.tsv'
#    selector = ModelSelector(petab_problem, model_specification_file)
#    result = selector.select('forward', 'AIC')


ms_file_text = '''ModelId\tSBML\tp1\tp2\tp3
m0\tsbml1.xml\t0;5;-\t0;-\t-
m1\tsbml1.xml\t0\t-\t-
m2\tsbml2.xml\t3;-\t-\t-
m3\tsbml2.xml\t-\t0;2\t-
'''

ms_file_text_unpacked = '''ModelId\tSBML\tp1\tp2\tp3
m0_0\tsbml1.xml\t0\t0\tnan
m0_1\tsbml1.xml\t0\tnan\tnan
m0_2\tsbml1.xml\t5\t0\tnan
m0_3\tsbml1.xml\t5\tnan\tnan
m0_4\tsbml1.xml\tnan\t0\tnan
m0_5\tsbml1.xml\tnan\tnan\tnan
m1_0\tsbml1.xml\t0\tnan\tnan
m2_0\tsbml2.xml\t3\tnan\tnan
m2_1\tsbml2.xml\tnan\tnan\tnan
m3_0\tsbml2.xml\tnan\t0\tnan
m3_1\tsbml2.xml\tnan\t2\tnan
'''


def test_ms_file_unpacking():
    ms_file = tempfile.NamedTemporaryFile(mode='r+', delete=False)
    with open(ms_file.name, 'w') as f:
        f.write(ms_file_text)
    ms_file_unpacked = unpack_file(ms_file.name)
    print(ms_file_unpacked.read())
    print(ms_file_text_unpacked)
    # assert str(ms_file_unpacked.read()) == ms_file_text_unpacked
    selector = ModelSelector(None, ms_file_unpacked.name)
    # assert False, selector.header


def test_get_next_step_candidates():
    ms_file = tempfile.NamedTemporaryFile(mode='r+', delete=False)
    with open(ms_file.name, 'w') as f:
        f.write(ms_file_text)
    selector = ModelSelector(None, ms_file.name)
    model_list = [model for model in selector.model_generator()]
    result, selection_history = selector.select('forward', 'AIC')
    print(selection_history)

