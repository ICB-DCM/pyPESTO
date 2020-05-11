"""
This is for testing the pypesto.model_selection.
"""
import tempfile
import numpy as np
from pypesto import PetabImporter, minimize
from pypesto.model_selection import (
    ModelSelector,
    ModelSelectionProblem,
    unpack_file,
    ForwardSelector
)
from pypesto.model_selection.constants import *

import petab
import math
import pytest
from typing import Dict, Set


EXAMPLE_YAML = 'doc/example/model_selection/example_modelSelection.yaml'
EXAMPLE_MODELS = ('doc/example/model_selection/'
                  'modelSelectionSpecification_example_modelSelection.tsv')

ms_file_text = f'''ModelId\tSBML\tp1\tp2\tp3
m0\tsbml1.xml\t0;5;-\t0;-\t-
m1\tsbml1.xml\t0\t-\t-
m2\tsbml2.xml\t3;-\t-\t-
m3\tsbml2.xml\t-\t0;2\t-
'''

ms_file_unpacked_text_expected = '''ModelId\tSBML\tp1\tp2\tp3
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


def test_unpack_file():
    ms_file = tempfile.NamedTemporaryFile(mode='r+', delete=False)
    with open(ms_file.name, 'w') as f:
        f.write(ms_file_text)
    ms_file_unpacked = unpack_file(ms_file.name)
    ms_file_unpacked_text = ms_file_unpacked.read()
    assert ms_file_unpacked_text == ms_file_unpacked_text_expected

# TODO check whether ModelSelectionProblem.n_estimated is correct. It's used
# to count the number of estimated parameters in the model, for use in the
# AIC calculation. However, this only counts estimated parameters in the
# parameters specified by the model specification file. There could be
# additional parameters, in the PEtab parameter file, that are specified to be
# estimated, but not included as a column in the model specification file.
# Hence, the AIC calculation would be incorrect.
# Perhaps not an issue for comparison of AIC between models, which is what AIC
# is used for, as the error would be that the AIC for all models would be less
# than the real AIC by some constant value (assuming all models use the same
# PEtab parameters file -- not the case when the YAML column is implemented).

def test_get_test_models():
    selector = ModelSelector(None, EXAMPLE_MODELS)
    #model0 = selector.new_direction_problem()

    forward_selector = ForwardSelector(
        None,
        selector.model_generator,
        None,
        selector.parameter_IDs,
        {}
    )
    forward_selector.initial_model = False

    model0 = forward_selector.new_direction_problem()
    result0 = forward_selector.get_test_models(model0)
    assert {model[MODEL_ID] for model in result0} == {'M5_0', 'M6_0', 'M7_0'}

    model_M5 = forward_selector.new_model_problem(
        next(model for model in result0 if model[MODEL_ID] == 'M5_0'),
        valid = False
    )
    result_M5 = forward_selector.get_test_models(model_M5)
    assert {model[MODEL_ID] for model in result_M5} == {'M2_0', 'M4_0'}

    model_M2 = forward_selector.new_model_problem(
        next(model for model in result_M5 if model[MODEL_ID] == 'M2_0'),
        valid = False
    )
    result_M2 = forward_selector.get_test_models(model_M2)
    assert {model[MODEL_ID] for model in result_M2} == {'M1_0'}


def test_relative_complexity_parameters():
    # 3-tuples: old parameter value, new parameter value, relative complexity
    # change
    tests = [
        (    0,     0,  0),
        (    0,     1,  0),
        (    0,    -5,  0),
        (    0, 'nan',  1),
        (    1,     0,  0),
        (    1,    -3,  0),
        (    1, 'nan',  1),
        ('nan',     0, -1),
        ('nan',    -2, -1),
        ('nan', 'nan',  0)
    ]

    selector = ForwardSelector(None, None, None, None, None)
    for test in tests:
        old, new, expected_complexity = (float(p) for p in test)
        assert selector.relative_complexity_parameters(old, new) == \
            expected_complexity


#from pypesto.model_selection import ForwardSelector
def test_relative_complexity_models_forward():
    # TODO in `model_selection.py`, consider how to handle models that are
    # equivalent and chosen as the next step candidates
    # TODO move MODEL_ID to constants file, change some headers in this list to
    # the constants
    # ignoring 'modelId', 'SBML' headers for now.
    headers = ['p1',         'p2',         'p3']
    model0_values  = [float('nan'), float(0),     float(3)]

    # List of 2-tuples, first value is expected complexity of model compared to
    # `model0`, second value is the model, in the same format as
    # `x = ModelSelector.model_generator(); x = x.values()`
    tests = [
        (0,     ['nan',     2,     3]),
        (1,     ['nan', 'nan',     3]),
        (1,     ['nan', 'nan',     0]),
        ('nan', [    0, 'nan',     3]),
        ('nan', [    2,     0, 'nan']),
    ]

    # `headers` is only `parameter_IDs` here. Normally also contains 'modelId'
    # and 'SBML'
    selector = ForwardSelector(None, None, None, headers, None)
    for expected_complexity_untyped, model_values_untyped in tests:
        expected_complexity = float(expected_complexity_untyped)
        model_values = [float(p) for p in model_values_untyped]
        if math.isnan(expected_complexity):
            assert math.isnan(
                selector.relative_complexity_models(
                    dict(zip(headers, model0_values)),
                    dict(zip(headers, model_values)))
            )
        else:
            assert selector.relative_complexity_models(
                dict(zip(headers, model0_values)),
                dict(zip(headers, model_values))
            ) == expected_complexity


def models_compared_with(model_ID0: str,
                         selection_history: Dict[str, Dict])-> Set[str]:
    model_IDs = set()
    for model_ID, model_info in selection_history.items():
        if model_info['compared_model_ID'] == model_ID0:
            model_IDs.add(model_ID)
    return model_IDs


def test_pipeline_forward():
    petab_problem = petab.Problem.from_yaml(EXAMPLE_YAML)

    selector = ModelSelector(petab_problem, EXAMPLE_MODELS)
    model_list = [model for model in selector.model_generator()]
    
    result, selection_history = selector.select('forward', 'AIC')
    assert models_compared_with(INITIAL_VIRTUAL_MODEL, selection_history) == \
        {'M5_0', 'M6_0', 'M7_0'}
    assert models_compared_with('M6_0', selection_history) == \
        {'M3_0', 'M4_0'}

    result, selection_history = selector.select('forward', 'AIC')
    # includes models compared to `INITIAL_VIRTUAL_MODEL` in first run, as
    # `selection_history` includes them (they were not retested)
    assert models_compared_with(INITIAL_VIRTUAL_MODEL, selection_history) == \
        {'M5_0', 'M6_0', 'M7_0', 'M2_0'}
    assert models_compared_with('M2_0', selection_history) == \
        {'M1_0'}

    with pytest.raises(Exception):
        # TODO ensure correct exception is raised?
        selector.select('forward', 'AIC')
    for s in selection_history:
        print(selection_history[s])


def test_pipeline_backward():
    petab_problem = petab.Problem.from_yaml(EXAMPLE_YAML)
    selector = ModelSelector(petab_problem, EXAMPLE_MODELS)
    model_list = [model for model in selector.model_generator()]

    result, selection_history = selector.select('backward', 'AIC')
    assert models_compared_with(INITIAL_VIRTUAL_MODEL, selection_history) == \
        {'M1_0'}
    assert models_compared_with('M1_0', selection_history) == \
        {'M2_0', 'M3_0', 'M4_0'}
    assert models_compared_with('M3_0', selection_history) == \
        {'M6_0', 'M7_0'}

    result, selection_history = selector.select('backward', 'AIC')
    # includes models compared to `INITIAL_VIRTUAL_MODEL` in first run, as
    # `selection_history` includes them (they were not retested)
    assert models_compared_with(INITIAL_VIRTUAL_MODEL, selection_history) == \
        {'M1_0', 'M5_0'}

    with pytest.raises(Exception):
        # TODO ensure correct exception is raised?
        selector.select('forward', 'AIC')
    for s in selection_history:
        print(selection_history[s])
