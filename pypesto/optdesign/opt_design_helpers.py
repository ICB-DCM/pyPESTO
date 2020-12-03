import numpy as np
from scipy.linalg import eigvalsh
from .design_problem import DesignProblem
from ..objective import AmiciObjective
from typing import Optional, Union, List, Iterable


# make this more readable
def get_hess(obj: AmiciObjective, x: np.ndarray):
    """
    computes the hessian for parameter x
    """
    hess = obj.get_hess(x)
    message = 'got hess via objective from specified parameters'

    if hess is None:
        raise RuntimeError("could not get hessian")

    # TODO properly handle this case
    if np.isnan(hess).any():
        print("hess is NaN.")
        # message = 'hessian is evaluated at the initial best parameter'
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


def get_eigmin_estimate(hess_additional: np.ndarray,
                        initial_result: dict):
    # computes an upper bound for the eigmin of FIM + V*V^T where V is a
    # matrix ie. FIM + V*V^T = FIM + v1*v1^T + v2*v2^T + ... + vn*vn^T
    # see: REFINED PERTURBATION BOUNDS FOR EIGENVALUES OF HERMITIAN AND
    # NON-HERMITIAN MATRICES - IPSEN, NADLER, Thm 2.1

    eigvals_FIM, eigvec_FIM = initial_result['eigen_decomposition']
    V = hess_additional
    new_col = []
    for col_ind in range(len(V[0])):
        new_col.append(np.array(
            [np.dot(eigvec_FIM[:, 1], V[:, col_ind]),
             np.dot(eigvec_FIM[:, 0], V[:, col_ind])]))

    add_new = np.vstack(new_col).transpose()

    eigmin_A_min = eigvals_FIM[0]
    eigmin_A_min_2 = eigvals_FIM[1]

    U = np.diag([eigmin_A_min_2, eigmin_A_min]) \
        + np.matmul(add_new, add_new.transpose())
    eigvals_U = np.linalg.eigvalsh(U)
    eigmin_U = np.min(eigvals_U)

    return eigmin_U


def get_ratio_estimate(eigmin_upper_bound: float,
                       initial_result: dict):
    eigvals_FIM, eigvec_FIM = initial_result['eigen_decomposition']
    eig_max = eigvals_FIM[-1]

    return eigmin_upper_bound / eig_max


# TODO restructure this part
# ideally with different functions for the single runs and for checking
# combinations
# and restructure for the faster implementations for some criteria
def get_design_result(design_problem: DesignProblem,
                      x: np.ndarray,
                      candidate: Optional[Union[list, dict]] = None,
                      hess: np.ndarray = None,
                      hess_additional: np.ndarray = None,
                      ) \
        -> dict:
    """

    Parameters
    ----------
    design_problem: the problem formulation
    x: the set of parameters for which criteria will be computed
    candidate: is a dict if we compute the result for a single experiment
               is a list of indices of we compute the result for a
               combination of experiments
               can be 'None', in this case, compute the initial results
               without adding new experiments
    hess: the hessian
    hess_additional: a matrix M st M*M^T will be added to the hessian
                     before any computation
    """
    message = None
    if isinstance(candidate, dict):
        dictionary = {'candidate': {'id': candidate['id']}}
    else:
        dictionary = {'candidate': candidate}

    if hess is None and candidate is None:
        hess, message = get_hess(obj=design_problem.problem.objective,
                                 x=x)
    # elif candidate is not None and hess is None:
    #     print("?")

    # this means we don't check hess, but
    # hess + hess_additional*hess_additional^T
    if hess_additional is not None and hess is not None:
        hess = hess + np.matmul(hess_additional, hess_additional.transpose())

    dictionary['x'] = x
    dictionary['hess'] = hess
    dictionary['message'] = message

    eigvals = get_eigvals(hess=hess)
    dictionary['eigvals'] = eigvals
    for criteria in design_problem.non_modified_criteria:
        dictionary[criteria] = get_criteria(criteria, hess, eigvals)

    if design_problem.modified_criteria:
        hess_modified = add_to_hess(hess=hess,
                                    const=design_problem.const_for_hess)
        eigvals_modified = get_eigvals(hess=hess_modified)
        for criteria in design_problem.modified_criteria:
            dictionary[''.join([criteria, '_modified'])] = \
                get_criteria(criteria, hess_modified, eigvals_modified)

    dictionary['constant_for_hessian'] = design_problem.const_for_hess
    return dictionary


def combinations_gen(elements: Iterable, length: int):
    """
    returns a generator which gives all possible combinations of length
    'length' from 'elements'
    """
    for i in range(len(elements)):
        if length == 1:
            yield [elements[i], ]
        else:
            for next in combinations_gen(elements[i + 1:len(elements)],
                                         length - 1):
                yield [elements[i], ] + next


def get_average_result_dict(list_of_dicts: List[dict]) \
        -> dict:
    """
    takes a list of dictionaries, as returned by 'get_design_result' and
    returns a dictionary of the average values for each criteria.
    For certain hardcoded keys, where the average doesn't make sense,
    write 'average' as entry to avoid confusion
    """
    ave_dict = {'candidate': list_of_dicts[0]['candidate'],
                'x': 'average',
                'hess': 'average',
                'message': 'average',
                'eigvals': 'average',
                'fim_addition': 'average',
                'fim_added': 'average',
                'constant_for_hessian': list_of_dicts[0][
                    'constant_for_hessian']}

    for key in list_of_dicts[0].keys() - ave_dict.keys():
        ave_dict[key] = sum(d[key] for d in list_of_dicts) / len(
            list_of_dicts)

    return ave_dict


def add_to_dict(dict_1: dict, dict_2: dict) \
        -> dict:
    """
    adds dict_1 and dict_2 together, except for certain keys, where it
    doesn't make sense
    """
    sum_dict = {'candidate': dict_1['candidate'],
                'x': 'average',
                'hess': 'average',
                'message': 'average',
                'eigvals': 'average',
                'fim_addition': 'average',
                'fim_added': 'average',
                'constant_for_hessian': dict_1['constant_for_hessian']}

    for key in dict_1.keys() - sum_dict.keys():
        sum_dict[key] = dict_1[key] + dict_2[key]

    return sum_dict


def divide_dict(dict: dict, div: float) \
        -> dict:
    """
    divides relevant values in dict by div
    changes the passed dict
    """
    no_keys = ['candidate', 'x', 'hess', 'message', 'eigvals', 'fim_addition',
               'fim_added', 'constant_for_hessian']
    for key in dict.keys() - no_keys:
        dict[key] = dict[key] / div

    return dict
