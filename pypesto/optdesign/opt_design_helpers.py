import numpy as np
from scipy.linalg import eigvalsh
from .design_problem import DesignProblem
from ..objective import AmiciObjective
from typing import Optional, Union, List


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


# TODO restructure this part
# ideally with different functions for the single runs and for checking
# combinations
# and restructure for the faster implementations for some criteria
def get_design_result(design_problem: DesignProblem,
                      x: np.ndarray,
                      candidate: Optional[Union[list, dict]] = None,
                      hess: np.ndarray = None,
                      hess_additional: np.ndarray = None,
                      initial_result: dict = None,
                      combi_runs: dict = None,
                      skip: bool = None,
                      criteria_min: dict = None
                      ) \
        -> dict:
    """

    Parameters
    ----------
    design_problem
    x
    candidate
    hess
    hess_additional
    initial_result
    combi_runs
    skip: bool, used right now to indicate if we want to use the fast
        implementation to compute eigmin, which only works when checking
        combinations
    criteria_min: dict which has all criteria as keys. for each criteria
        save the lowest value of the n_save_combi_result that we save
        -> used to check if the upper bound on eigmin is lower
        -> then skip the computation

    Returns
    -------

    """
    message = None
    dict = {'candidate': candidate}

    # used when we check different combinations in the end thus checking
    # different FIMs
    if hess is not None:
        pass
    # check if the forward simulation failed
    # also used in the first initial check (hence 'if candidate is None')
    elif candidate is None:  # or ~np.isnan(
        # design_problem.petab_problem.measurement_df.measurement).any()

        hess, message = get_hess(obj=design_problem.problem.objective,
                                 x=x, fallback=design_problem.initial_x)
    else:
        message = "Simulation failed. Simulated measurement is NaN"
        print("Simulation failed. Simulated measurement is NaN")
        hess = None

    # this means we don't check hess, but
    # hess + hess_additional*hess_additional^T
    if hess_additional is not None:
        hess = hess + np.matmul(hess_additional, hess_additional.transpose())

    dict['x'] = x
    dict['hess'] = hess
    dict['message'] = message

    # use matrix-determinant-lemma
    # 'hess_additional is not None' to exclude the evaluataion of the initial
    # result
    # 'skip is only True when we check combinations

    # TODO this is only for testing right now and has to be completely reworked
    if (design_problem.criteria_list == ['det'] or
        set(design_problem.criteria_list) == {'eigmin', 'det'} or
        design_problem.criteria_list == ['eigmin']) \
            and hess_additional is not None \
            and skip is True:

        if 'det' in design_problem.criteria_list:
            dict['det'] = np.linalg.det(
                np.eye(len(hess_additional[0])) + np.matmul(
                    hess_additional.transpose(),
                    np.matmul(initial_result['hess_inv'], hess_additional))) \
                          * initial_result['det']

        if 'eigmin' in design_problem.criteria_list:
            upper_bound = get_eigmin_estimate(hess_additional=hess_additional,
                                              initial_result=initial_result)
            try:
                smallest_best_value = criteria_min['eigmin']
            except KeyError:
                smallest_best_value = np.infty
            if len(combi_runs['eigmin']) < \
                    design_problem.n_save_combi_result \
                    or upper_bound >= smallest_best_value:
                # compute only the smallest eigenvalue

                # not sure if actually faster
                eigmin = eigvalsh(hess, eigvals=(0, 0))[0]
                dict['eigmin'] = eigmin

            else:
                dict['eigmin'] = np.NINF
                pass

    else:
        # 'old' method where we compute all eigenvalues and then each criteria
        eigvals = get_eigvals(hess=hess)
        dict['eigvals'] = eigvals
        for criteria in design_problem.non_modified_criteria:
            dict[criteria] = get_criteria(criteria, hess, eigvals)

    if design_problem.modified_criteria:
        if (design_problem.criteria_list == ['det'] or
            set(design_problem.criteria_list) == {'eigmin', 'det'} or
            design_problem.criteria_list == ['eigmin']) \
                and hess_additional is not None \
                and skip is True:
            if 'det' in design_problem.criteria_list:
                dict['det_modified'] = np.linalg.det(
                    np.eye(len(hess_additional[
                                   0]))
                    + np.matmul(
                        hess_additional.transpose(),
                        np.matmul(initial_result['hess_inv_modified'],
                                  hess_additional))) * initial_result[
                                           'det_modified']

            if 'eigmin' in design_problem.criteria_list:
                dict['eigmin_modified'] = dict['eigmin'] + \
                                          design_problem.const_for_hess
        else:
            hess_modified = add_to_hess(hess=hess,
                                        const=design_problem.const_for_hess)
            eigvals_modified = get_eigvals(hess=hess_modified)
            for criteria in design_problem.modified_criteria:
                dict[''.join([criteria, '_modified'])] = \
                    get_criteria(criteria, hess_modified, eigvals_modified)

    # dict['result'] = result
    dict['constant_for_hessian'] = design_problem.const_for_hess
    return dict


def combinations_gen(elements, length: int):
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
    Parameters
    ----------
    list_of_dicts

    Returns
    -------

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
