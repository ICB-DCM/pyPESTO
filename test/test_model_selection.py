"""
Test pypesto.model_selection.
"""
import math
import numpy as np
import pandas as pd
import pytest
import tempfile
from typing import Dict, List, Set

import petab
from pypesto import PetabImporter

from pypesto.model_selection import (
    ForwardSelector,
    ModelSelectionProblem,
    ModelSelector,
    ModelSelectorMethod,
    row2problem,
    unpack_file,
)

from pypesto.model_selection.constants import (
    COMPARED_MODEL_ID,
    ESTIMATE_SYMBOL_INTERNAL,
    ESTIMATE_SYMBOL_UI,
    INITIAL_VIRTUAL_MODEL,
    MODEL_ID,
    NOT_PARAMETERS,
)

EXAMPLE_YAML = 'doc/example/model_selection/example_modelSelection.yaml'
EXAMPLE_MODELS = ('doc/example/model_selection/'
                  'modelSelectionSpecification_example_modelSelection.tsv')

ms_file_text = '''ModelId\tSBML\tp1\tp2\tp3
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


def get_test_model_tree_flat(
        model: ModelSelectionProblem,
        selector: ModelSelectorMethod,
        expected_test_model_ids: Dict[str, Set[str]],
        initial_virtual_model: bool = False
) -> List[str]:
    test_model_tree_flat = [model.model_id]
    result = selector.get_test_models(model)
    if initial_virtual_model:
        selector.initial_model = False
    assert {m[MODEL_ID] for m in result} == \
        expected_test_model_ids[model.model_id]
    for test_model_row in result:
        test_model = selector.new_model_problem(
            test_model_row,
            compared_model_id=model.model_id,  # unnecessary?
            valid=False
        )
        test_model_tree_flat += get_test_model_tree_flat(
            test_model,
            selector,
            expected_test_model_ids
        )
    return sorted(test_model_tree_flat)


def directional_get_test_models(expected_test_model_ids: Dict[str, Set[str]],
                                expected_test_model_tree_flat: List[str],
                                reverse: bool):
    model_selector = ModelSelector(None, EXAMPLE_MODELS)
    selector = ForwardSelector(
        None,
        model_selector.model_generator,
        None,
        model_selector.parameter_ids,
        {},
        None,
        reverse,
        False,
        False,
        None,
        0
    )
    initial_virtual_model = selector.new_direction_problem()
    test_model_tree_flat = get_test_model_tree_flat(initial_virtual_model,
                                                    selector,
                                                    expected_test_model_ids,
                                                    initial_virtual_model=True)
    print(test_model_tree_flat)
    print(expected_test_model_tree_flat)
    assert test_model_tree_flat == expected_test_model_tree_flat


def test_forward_selector_get_test_models():
    # TODO redo with networkx
    # get_test_model(model_id) for all model IDs as model_id
    expected_test_model_ids = {
        INITIAL_VIRTUAL_MODEL: {'M1_0_0'},
        'M1_0_0': {'M1_1_0', 'M1_2_0', 'M1_3_0'},
        'M1_1_0': {'M1_4_0', 'M1_5_0'},
        'M1_2_0': {'M1_4_0', 'M1_6_0'},
        'M1_3_0': {'M1_5_0', 'M1_6_0'},
        'M1_4_0': {'M1_7_0'},
        'M1_5_0': {'M1_7_0'},
        'M1_6_0': {'M1_7_0'},
        'M1_7_0': set(),
    }

    # returned test model IDs for a recursive call to get_test_model()
    # starting at INITIAL_VIRTUAL_MODEL. e.g.: sort and flatten the tree of
    # model IDs described by expected_test_model_ids, where the base of the
    # tree is M1_0_0.
    expected_test_model_tree_flat = sorted([
        INITIAL_VIRTUAL_MODEL,
        # INITIAL_VIRTUAL_MODEL
        'M1_0_0',
        # M1_0_0
        'M1_1_0',
        'M1_2_0',
        'M1_3_0',
        # M1_1_0
        'M1_4_0',
        'M1_5_0',
        # M1_2_0
        'M1_4_0',
        'M1_6_0',
        # M1_3_0
        'M1_5_0',
        'M1_6_0',
        # M1_4_0 from M1_1_0
        'M1_7_0',
        # M1_4_0 from M1_2_0
        'M1_7_0',
        # M1_5_0 from M1_1_0
        'M1_7_0',
        # M1_5_0 from M1_3_0
        'M1_7_0',
        # M1_6_0 from M1_2_0
        'M1_7_0',
        # M1_6_0 from M1_6_0
        'M1_7_0',
    ])

    directional_get_test_models(expected_test_model_ids,
                                expected_test_model_tree_flat,
                                reverse=False)


def test_backward_selector_get_test_models():
    # get_test_model(model_id) for all model IDs as model_id
    expected_test_model_ids = {
        INITIAL_VIRTUAL_MODEL: {'M1_7_0'},
        'M1_0_0': set(),
        'M1_1_0': {'M1_0_0'},
        'M1_2_0': {'M1_0_0'},
        'M1_3_0': {'M1_0_0'},
        'M1_4_0': {'M1_1_0', 'M1_2_0'},
        'M1_5_0': {'M1_1_0', 'M1_3_0'},
        'M1_6_0': {'M1_2_0', 'M1_3_0'},
        'M1_7_0': {'M1_4_0', 'M1_5_0', 'M1_6_0'},
    }

    # returned test model IDs for a recursive call to get_test_model()
    # starting at INITIAL_VIRTUAL_MODEL. e.g.: sort and flatten the tree of
    # model IDs described by expected_test_model_ids, where the base of the
    # tree is M1_0_0.
    expected_test_model_tree_flat = sorted([
        INITIAL_VIRTUAL_MODEL,
        # INITIAL_VIRTUAL_MODEL
        'M1_7_0',
        # M1_7_0
        'M1_4_0',
        'M1_5_0',
        'M1_6_0',
        # M1_4_0
        'M1_1_0',
        'M1_2_0',
        # M1_5_0
        'M1_1_0',
        'M1_3_0',
        # M1_6_0
        'M1_2_0',
        'M1_3_0',
        # M1_1_0 from M1_4_0
        'M1_0_0',
        # M1_1_0 from M1_5_0
        'M1_0_0',
        # M1_2_0 from M1_4_0
        'M1_0_0',
        # M1_2_0 from M1_6_0
        'M1_0_0',
        # M1_3_0 from M1_5_0
        'M1_0_0',
        # M1_3_0 from M1_6_0
        'M1_0_0',
    ])

    directional_get_test_models(expected_test_model_ids,
                                expected_test_model_tree_flat,
                                reverse=True)


def test_relative_complexity_parameters():
    # 3-tuples: old parameter value, new parameter value, relative complexity
    # change
    tests = [
        (0,         0,  0),
        (0,         1,  0),
        (0,        -5,  0),
        (0,     'nan',  1),
        (1,         0,  0),
        (1,        -3,  0),
        (1,     'nan',  1),
        ('nan',     0, -1),
        ('nan',    -2, -1),
        ('nan', 'nan',  0)
    ]

    # selector = ForwardSelector(None, None, None, None, None)
    selector = ForwardSelector(*([None]*11))
    for test in tests:
        old, new, expected_complexity = (float(p) for p in test)
        assert selector.relative_complexity_parameters(old, new) == \
            expected_complexity


# from pypesto.model_selection import ForwardSelector
def test_relative_complexity_models_forward():
    # TODO in `model_selection.py`, consider how to handle models that are
    # equivalent and chosen as the next step candidates
    # TODO move MODEL_ID to constants file, change some headers in this list to
    # the constants
    # ignoring 'modelId', 'SBML' headers for now.
    headers = ['p1', 'p2', 'p3']
    model0_values = [float('nan'), float(0), float(3)]

    # List of 2-tuples, first value is expected complexity of model compared to
    # `model0`, second value is the model, in the same format as
    # `x = ModelSelector.model_generator(); x = x.values()`
    tests_strict_true = [
        ('nan', ['nan',     2,     3]),
        (1,     ['nan', 'nan',     3]),
        (1,     ['nan', 'nan',     0]),
        ('nan', [0,     'nan',     3]),
        ('nan', [2,         0, 'nan']),
    ]

    # TODO implement tests, and code for self.strict in ForwardSelector
    # tests_strict_false = [
    #     (0,     ['nan',     2,     3]),
    #     (1,     ['nan', 'nan',     3]),
    #     (1,     ['nan', 'nan',     0]),
    #     ('nan', [    0, 'nan',     3]),
    #     ('nan', [    2,     0, 'nan']),
    # ]

    # `headers` is only `parameter_ids` here. Normally also contains 'modelId'
    # and 'SBML'
    selector = ForwardSelector(*([None]*3), headers, *([None]*7))
    selector.reverse = False
    for expected_complexity_untyped, model_values_untyped in tests_strict_true:
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


def models_compared_with(model_id0: str,
                         selection_history: Dict[str, Dict]) -> Set[str]:
    model_ids = set()
    for model_id, model_info in selection_history.items():
        if model_info[COMPARED_MODEL_ID] == model_id0:
            model_ids.add(model_id)
    return model_ids


def test_pipeline_forward():
    # TODO rewrite with networkx
    # TODO test may fail, depending on pypesto.minimise outcome for each model.
    # rewrite such that tests are predictable
    petab_problem = petab.Problem.from_yaml(EXAMPLE_YAML)
    selector = ModelSelector(petab_problem, EXAMPLE_MODELS)
    # model_list = [model for model in selector.model_generator()]

    selected_models, _, selection_history = selector.select('forward', 'AIC')
    assert models_compared_with(INITIAL_VIRTUAL_MODEL, selection_history) == \
        {'M1_0_0'}
    assert models_compared_with('M1_0_0', selection_history) == \
        {'M1_1_0', 'M1_2_0', 'M1_3_0'}
    assert models_compared_with('M1_1_0', selection_history) == \
        {'M1_4_0', 'M1_5_0'}

    selected_models, local_selection_history, selection_history = \
        selector.select('forward', 'AIC')
    # includes models compared to `INITIAL_VIRTUAL_MODEL` in first run, as
    # `selection_history` includes them (they were not retested)
    assert models_compared_with(INITIAL_VIRTUAL_MODEL, selection_history) == \
        {'M1_0_0', 'M1_6_0'}
    assert models_compared_with('M1_6_0', selection_history) == \
        {'M1_7_0'}

    with pytest.raises(Exception):
        # TODO ensure correct exception is raised?
        selector.select('forward', 'AIC')
    for s in selection_history:
        print(selection_history[s])


def test_pipeline_backward():
    petab_problem = petab.Problem.from_yaml(EXAMPLE_YAML)
    selector = ModelSelector(petab_problem, EXAMPLE_MODELS)
    # model_list = [model for model in selector.model_generator()]

    selected_models, _, selection_history = selector.select('backward', 'AIC')
    assert models_compared_with(INITIAL_VIRTUAL_MODEL, selection_history) == \
        {'M1_7_0'}
    assert models_compared_with('M1_7_0', selection_history) == \
        {'M1_4_0', 'M1_5_0', 'M1_6_0'}
    assert models_compared_with('M1_6_0', selection_history) == \
        {'M1_2_0', 'M1_3_0'}
    assert models_compared_with('M1_3_0', selection_history) == \
        {'M1_0_0'}

    selected_models, local_selection_history, selection_history = \
        selector.select('backward', 'AIC')
    # includes models compared to `INITIAL_VIRTUAL_MODEL` in first run, as
    # `selection_history` includes them (they were not retested)
    assert models_compared_with(INITIAL_VIRTUAL_MODEL, selection_history) == \
        {'M1_7_0', 'M1_1_0'}

    with pytest.raises(Exception):
        # TODO ensure correct exception is raised?
        selector.select('backward', 'AIC')
    for s in selection_history:
        print(selection_history[s])


def test_row2problem_yaml_string():
    petab_problem = petab.Problem.from_yaml(EXAMPLE_YAML)
    importer = PetabImporter(petab_problem)
    obj = importer.create_objective()
    expected_problem = importer.create_problem(obj)

    problem = row2problem({}, EXAMPLE_YAML)
    assert (problem.x_names == expected_problem.x_names and
            problem.x_free_indices == expected_problem.x_free_indices and
            problem.x_fixed_indices == expected_problem.x_fixed_indices and
            problem.x_fixed_vals == expected_problem.x_fixed_vals)


def test_row2problem_setting_pars():
    models_df = pd.read_csv(EXAMPLE_MODELS, sep='\t')
    for _, row in models_df.iterrows():
        for key, val in row.items():
            if key not in NOT_PARAMETERS:
                row[key] = float(ESTIMATE_SYMBOL_INTERNAL) \
                    if val == ESTIMATE_SYMBOL_UI else float(val)

        problem = row2problem(dict(row), EXAMPLE_YAML)

        petab_problem = petab.Problem.from_yaml(EXAMPLE_YAML)
        parameter_df_x_ids = tuple(petab_problem.parameter_df.index)

        for key in NOT_PARAMETERS:
            row.pop(key)

        x_free_indices = []
        x_fixed_indices = []
        x_fixed_vals = []
        for x_id, x_val in row.items():
            if np.isnan(x_val):
                x_free_indices.append(parameter_df_x_ids.index(x_id))
            else:
                x_fixed_indices.append(parameter_df_x_ids.index(x_id))
                x_fixed_vals.append(x_val)

        # Need to remove parameters that are in the PEtab parameters table
        # but not in the model spec file.
        mask = [True if x_id in row.keys() else False
                for x_id in parameter_df_x_ids]
        test_x_free_indices = [x_index for x_index in problem.x_free_indices
                               if mask[x_index]]
        test_x_fixed_indices = [x_index for x_index in problem.x_fixed_indices
                                if mask[x_index]]
        test_x_fixed_vals = [x_index for index, x_index
                             in enumerate(problem.x_fixed_vals)
                             if mask[problem.x_fixed_indices[index]]]
        assert (test_x_free_indices == x_free_indices and
                test_x_fixed_indices == x_fixed_indices and
                test_x_fixed_vals == x_fixed_vals)


@pytest.mark.skip
def test_custom_initial_model():
    pass
