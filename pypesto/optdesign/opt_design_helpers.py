import numpy as np
from scipy.linalg import eigvalsh
from .design_problem import DesignProblem
from ..objective import AmiciObjective
from typing import Optional, Union


# make this more readable
def get_hess(obj: AmiciObjective, x: np.ndarray,
             fallback: np.ndarray):
    hess = obj.get_hess(x)
    message = 'got hess via objective from specified parameters'

    if hess is None:
        raise RuntimeError("could not get hessian")

    # TODO properly handle this case
    if np.isnan(hess).any():
        print("hess is NaN.")
        message = 'hessian is evaluated at the initial best parameter'
        hess = obj.get_hess(fallback)
        if np.isnan(hess).any():
            hess = None
    return hess, message


def get_eigvals(hess: np.ndarray):
    if hess is None:
        return None
    v = eigvalsh(hess)
    return v


def add_to_hess(hess: np.ndarray, const: float):
    if hess is None:
        return None
    modified_hess = hess + const * np.eye(len(hess))
    return modified_hess


# TODO put the treshhold in design_problem
def get_criteria(criteria: str, hess: np.ndarray, eigvals: np.ndarray,
                 tresh: float = 10 ** (-4)):
    if hess is None:
        return None

    if criteria == 'det':
        value = np.prod(eigvals)
    elif criteria == 'trace':
        value = np.sum(eigvals)
    elif criteria == 'rank':
        value = np.linalg.matrix_rank(hess)
    elif criteria == 'trace_log':
        log_eigvals = np.log(np.absolute(eigvals))
        value = np.sum(log_eigvals)
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
                      x: np.ndarray,
                      candidate: Optional[Union[list, dict]] = None,
                      hess: np.ndarray = None
                      ):
    message = None
    dict = {'candidate': candidate}

    # used when we check different combinations in the end thus checking
    # different FIMs
    if hess is not None:
        pass
    # check if the forward simulation failed
    # also used in the first initial check (hence 'if candidate is None')
    elif candidate is None: # or ~np.isnan(
            # design_problem.petab_problem.measurement_df.measurement).any()

        hess, message = get_hess(obj=design_problem.problem.objective,
                                 x=x, fallback=design_problem.initial_x)
    else:
        message = "Simulation failed. Simulated measurement is NaN"
        print("Simulation failed. Simulated measurement is NaN")
        hess = None
    eigvals = get_eigvals(hess=hess)
    dict['x'] = x
    dict['hess'] = hess
    dict['message'] = message
    dict['eigvals'] = eigvals
    for criteria in design_problem.criteria_list:
        dict[criteria] = get_criteria(criteria, hess, eigvals)

    if design_problem.const_for_hess:
        hess_modified = add_to_hess(hess=hess,
                                    const=design_problem.const_for_hess)
        eigvals_modified = get_eigvals(hess=hess_modified)
        for criteria in design_problem.criteria_list:
            dict[''.join([criteria, '_modified'])] = \
                get_criteria(criteria, hess_modified, eigvals_modified)

    if design_problem.profiles:
        raise NotImplementedError(
            "profile based criteria are not supported yet")
        # problem = design_problem.problem
        # result = get_profiles(result=design_problem.result, problem=problem)
        # dict['conf_interval'] = get_conf_intervals(result=result,
        # problem=problem)
        # dict['conf_interval_criteria'] = get_conf_interval_criteria(
        # result=result, problem=problem)

    # dict['result'] = result
    dict['constant_for_hessian'] = design_problem.const_for_hess
    return dict


def combinations_gen(elements, length):
    for i in range(len(elements)):
        if length == 1:
            yield [elements[i], ]
        else:
            for next in combinations_gen(elements[i + 1:len(elements)],
                                         length - 1):
                yield [elements[i], ] + next
