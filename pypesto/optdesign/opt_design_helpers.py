import numpy as np
from ..petab import PetabImporter
import amici
from .design_problem import DesignProblem
from ..objective import AmiciObjective
from ..result import Result
from copy import deepcopy
from typing import Optional


# make this more readable
def get_hess(obj: AmiciObjective, result: Optional[Result] = None,
             x: Optional[np.ndarray] = None):
    # if both result and x are passed, use x
    # get hess from function call instead of reading it from result
    if result is None and x is None:
        raise RuntimeError("you have to pass either the result-object or the "
                           "exact parameters where you want to "
                           "evaluate the hessian")
    hess = None
    message = None

    if x is not None:
        hess = obj.get_hess(x)
        message = 'got hess via objective from specified parameters'
    elif result is not None:
        hess = obj.get_hess(result.optimize_result.as_list()[0]['x'])

    if hess is None:
        raise RuntimeError("could not get hessian")

    # TODO properly handle this case
    if np.isnan(hess[0][0]):
        print("hess is NaN.")
        message = 'hessian is evaluated at result.problem.x_guesses[0]'
        hess = obj.get_hess(result.problem.x_guesses[0])

    return hess, message


def get_eigvals(hess: np.ndarray):
    v = np.linalg.eigvals(hess)
    return v


def get_best_parameters(number: int, result: Result):
    dicts = result.optimize_result.as_list(['x'])[0:number]
    return [d['x'] for d in dicts]


# TODO which settings to choose here?
def update_pypesto_from_petab(design_problem: DesignProblem):
    importer = PetabImporter(design_problem.petab_problem)
    obj = importer.create_objective(model=design_problem.model)
    obj.amici_solver.setSensitivityMethod(amici.SensitivityMethod.forward)
    obj.amici_solver.setAbsoluteToleranceFSA(1e-7)
    obj.amici_solver.setRelativeToleranceFSA(1e-7)
    obj.amici_solver.setAbsoluteTolerance(1e-7)
    obj.amici_solver.setRelativeTolerance(1e-7)
    obj.amici_model.setSteadyStateSensitivityMode(
        amici.SteadyStateSensitivityMode.simulationFSA)
    problem = importer.create_problem(obj)
    design_problem.problem = problem
    return design_problem


def add_to_hess(hess: np.ndarray, const: float):
    modified_hess = hess + const * np.eye(len(hess))
    return modified_hess


# TODO put the treshhold in design_problem
def get_criteria(criteria: str, hess: np.ndarray, eigvals: np.ndarray,
                 tresh: float = 10 ** (-4)):
    if criteria == 'det':
        value = np.linalg.det(hess)
    elif criteria == 'trace':
        value = np.sum(eigvals)
    elif criteria == 'rank':
        value = np.linalg.matrix_rank(hess)
    elif criteria == 'trace_log':
        value = 3
    elif criteria == 'ratio':
        value = np.amin(eigvals) / np.amax(eigvals)
    elif criteria == 'eigmin':
        value = np.amin(eigvals)
    elif criteria == 'number_good_eigvals':
        value = sum(1 for i in eigvals if i > tresh)
    else:
        raise Warning("can't find criteria specified in criteria_list")

    return value


# TODO split this up into subparts
# should the criteria be implemented as properties ?
def get_design_result(design_problem: DesignProblem,
                      candidate: Optional[dict] = None,
                      fn: Optional[str] = None,
                      result: Optional[Result] = None,
                      x: Optional[np.ndarray] = None):
    dict = {'candidate': candidate, 'fn': fn,
            'petab_problem': deepcopy(design_problem.petab_problem)}
    hess, message = get_hess(obj=design_problem.problem.objective,
                             result=design_problem.result, x=x)
    eigvals = get_eigvals(hess=hess)
    dict['hess'] = hess
    dict['eigvals'] = eigvals
    for criteria in design_problem.criteria_list:
        dict[criteria] = get_criteria(criteria, hess, eigvals)

    if design_problem.const_for_hess:
        hess_modified = add_to_hess(hess=hess,
                                    const=design_problem.const_for_hess)
        eigvals_modified = get_eigvals(hess=hess_modified)
        dict['hess_modified'] = hess_modified
        dict['eigvals_modified'] = eigvals_modified
        for criteria in design_problem.criteria_list:
            dict[criteria + '_modified'] = get_criteria(criteria,
                                                        hess_modified,
                                                        eigvals_modified)

    if design_problem.profiles:
        raise NotImplementedError(
            "profile based criteria are not supported yet")
        # problem = design_problem.problem
        # result = get_profiles(result=design_problem.result, problem=problem)
        # dict['conf_interval'] = get_conf_intervals(result=result,
        # problem=problem)
        # dict['conf_interval_criteria'] = get_conf_interval_criteria(
        # result=result, problem=problem)

    dict['result'] = result
    dict['constant_for_hessian'] = design_problem.const_for_hess
    return dict


# profiles
"""
def get_profiles(result: Result, problem: Problem, result_index: int = 0):
    optimizer = ScipyOptimizer()
    result = pypesto.profile.parameter_profile(
        problem=problem,
        result=result,
        optimizer=optimizer,
        profile_index=np.ones(len(problem.x_free_indices)),
        result_index=result_index)
    return result


def get_conf_intervals(result: Result, problem: Problem, log: bool = True):
    # Do we want to include sigma_length ?
    # since we add artificial noise which may influence it ?
    lengths = np.nan * np.ones(len(problem.x_free_indices))
    for i in range(len(problem.x_free_indices)):
        xmaxlog = result.profile_result.list[0][i]['x_path'][i][-1]
        xminlog = result.profile_result.list[0][i]['x_path'][i][0]
        if log:
            lengths[i] = xmaxlog - xminlog
        else:
            lengths[i] = 10 ** xmaxlog - 10 ** xminlog
    return lengths


def get_conf_interval_criteria(result: Result, problem: Problem):
    # add lengths of confidence intervals FOR LOG PARAMETERS
    lengths = get_conf_intervals(result, problem, log=True)
    return np.sum(lengths)


def plot_profile(result: Result, problem: Problem, obj: AmiciObjective,
                 index: int = 0):
    ref = pypesto.visualize.create_references(
        x=result.optimize_result.as_list(['x'])[index]['x'],
        fval=obj(result.optimize_result.as_list(['x'])[index]['x']))
    pypesto.visualize.profiles(result, profile_indices=range(
        len(problem.x_free_indices)),
                               reference=ref, profile_list_ids=0)
"""
