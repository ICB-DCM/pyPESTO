"""
This is for testing the pypesto.optdesign.
"""
import os
import numpy as np
import pypesto
import pypesto.optdesign as op
import pandas as pd
from pypesto.optdesign.design_problem import DesignProblem
from pypesto.optdesign.opt_design_helpers import get_design_result, \
    get_hess, get_eigvals, add_to_hess, get_criteria, combinations_gen, \
    add_to_dict, divide_dict, get_average_result_dict
import scipy.optimize as so
from typing import Generator


def load_design_problem():
    example_name = 'conversion_reaction'
    model_name = 'conversion_reaction'
    YAML_PATH = os.path.join('doc', 'example', example_name,
                             model_name + '.yaml')

    importer = pypesto.petab.PetabImporter.from_yaml(YAML_PATH)
    problem = importer.create_problem()
    experiment_list = load_experiment_list(n_experiments=10)
    design_problem = op.DesignProblem(experiment_list=experiment_list,
                                      model=problem.objective.amici_model,
                                      problem=problem,
                                      petab_problem=importer.petab_problem,
                                      initial_x=np.log([0.8, 0.6]))
    return design_problem


def load_result(design_problem):
    optimizer = pypesto.optimize.ScipyOptimizer(
        'L-BFGS-B')  # options={'maxiter': 50}
    result = pypesto.optimize.minimize(
        problem=design_problem.problem, optimizer=optimizer, n_starts=1,
    )
    return result


def load_experiment_list(n_experiments):
    n_experiments = n_experiments
    experiment_list = []
    grid = np.linspace(100, 110, n_experiments)
    for index, time in enumerate(grid):
        dict = {'id': index,
                'condition_df': None,
                'observable_df': None,
                'measurement_df': pd.DataFrame(
                    [['obs_a', 'c0', time, float('NaN'),
                      0.02 + index * 0.001]],
                    columns=pd.Index(['observableId', 'simulationConditionId',
                                      'time', 'measurement',
                                      'noiseParameters']))}

        experiment_list.append(dict)
    return experiment_list


def test_design_problem():
    design_problem = load_design_problem()
    assert isinstance(design_problem, DesignProblem)


def test_run_exp_design():
    design_problem = load_design_problem()

    res = op.run_exp_design(design_problem)

    for criteria in design_problem.criteria_list:
        if criteria not in ['rank', 'rank_modified', 'number_good_eigvals',
                            'number_good_eigvals_modified']:
            assert res.get_best_conditions(criteria=criteria, n_best=10)[1] \
                   == \
                   list(range(0, 10))


def test_hess():
    x = [0, 1]
    obj = pypesto.Objective(fun=so.rosen, grad=so.rosen_der,
                            hess=so.rosen_hess)
    exact_hess = so.rosen_hess([0, 1])
    opt_version = get_hess(obj=obj,
                           x=x)
    assert np.all(exact_hess == opt_version[0])

    exact_added = np.array([[-395, 0], [0, 203]])
    added = add_to_hess(so.rosen_hess(x), 3)
    assert np.all(exact_added == added)


def test_criteria():
    criteria_list = ['det', 'trace', 'rank', 'trace_log', 'ratio', 'eigmin',
                     'number_good_eigvals']
    tresh = 2
    mat = np.array([[1, 2], [2, 1]])
    exact_eigvals = np.array([-1, 3])
    eigvals = get_eigvals(mat)
    assert np.all(exact_eigvals == eigvals)

    solution = [-3, 2, 2, np.log(1) + np.log(3), -1 / 3, -1, 1]
    values = []
    for criteria in criteria_list:
        values.append(get_criteria(criteria, mat, eigvals, tresh))

    assert np.all(solution == values)


def test_combinations_gen():
    gen = combinations_gen([1, 2, 3], 2)
    assert isinstance(gen, Generator)
    assert np.all(list(gen) == [[1, 2], [1, 3], [2, 3]])


def test_dict_operations():
    dict_1 = {'candidate': None,
              'constant_for_hessian': None,
              'key1': 42,
              'key2': 1729}
    dict_2 = {'candidate': None,
              'constant_for_hessian': None,
              'key1': 42,
              'key2': 57}

    sum = add_to_dict(dict_1, dict_2)
    assert (sum['candidate'] is None and sum['constant_for_hessian'] is None)
    for key in dict_1:
        if key not in ['key1', 'key2', 'candidate', 'constant_for_hessian']:
            assert dict_1[key] == 'average'
    assert (sum['key1'] == 42 + 42 and sum['key2'] == 1729 + 57)

    divide_dict(sum, 2)
    div = sum
    assert (div['candidate'] is None and div['constant_for_hessian'] is None)
    for key in dict_1:
        if key not in ['key1', 'key2', 'candidate', 'constant_for_hessian']:
            assert dict_1[key] == 'average'
    assert (div['key1'] == (42 + 42) / 2 and div['key2'] == (1729 + 57) / 2)

    average = get_average_result_dict([dict_1, dict_2])
    assert div == average


def test_get_design_result_general():
    design_problem = load_design_problem()
    result = load_result(design_problem)
    x = result.optimize_result.get_for_key('x')[0]
    candidate = design_problem.experiment_list[0]
    mat = np.array([[1, 2], [2, 1]])
    res = get_design_result(design_problem, x, candidate, mat,
                            hess_additional=None,
                            )
    assert isinstance(res, dict)

    res2 = get_design_result(design_problem, x, candidate, mat,
                             hess_additional=np.array([[1], [1]]),
                             )
    assert isinstance(res2, dict)
    np.all(np.isclose(res2['eigvals'], [-1., 5.], atol=1e-12))
    for key in res:
        if key not in ['candidate', 'x', 'hess', 'message', 'eigvals']:
            assert isinstance(res[key], (int, float, np.int64))
            assert isinstance(res2[key], (int, float, np.int64))
