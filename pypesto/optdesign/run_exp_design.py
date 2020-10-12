from .design_problem import DesignProblem
from .result import DesignResult
from .opt_design_helpers import get_design_result, get_average_result_dict
from .change_dataframe import get_fim_addition, get_derivatives
import numpy as np
from typing import Iterable


def single_design_algo(design_result: DesignResult,
                       x: Iterable[float],
                       index: int
                       ) -> DesignResult:
    """
    Algorithm to find the single best condition to be added.
    For all candidates in design_problem.experiment_list measurements are
    simulated, added to the problem and a multi-start optimization is run to
    find the new optimum.
    The FIM at this new optimum is computed and it's eigenvalues are used to
    compute criteria values.
    The results are saved in 'design_result.single_runs'


    Parameters
    ----------
    design_result:
        result object for an experimental design setting
    x: the single set of parameters that will be used
    index: the index of the single x in the list design_problem.initial_x


    Returns
    -------
    design_result:
        the altered design_result
    """
    design_problem = design_result.design_problem
    initial_result = get_design_result(design_problem=design_problem,
                                       candidate=None,
                                       x=x)
    # inverse of the FIM might be used for the computation of the
    # det
    # TODO for faster det computation, handle case of FIM is not invertible
    try:
        initial_result['hess_inv'] = np.linalg.inv(initial_result['hess'])
    except np.linalg.LinAlgError:
        print("can't invert FIM")

    if design_problem.modified_criteria:
        hess_mod = initial_result['hess'] + \
                   design_problem.const_for_hess * np.eye(
            len(initial_result['hess']))
        initial_result['hess_inv_modified'] = np.linalg.inv(hess_mod)

    # TODO can be made optional
    # eigenvalue decomposition (including eigenvectors as this is needed for
    # eigmin estimates
    initial_result['eigen_decomposition'] = np.linalg.eigh(
        initial_result['hess'])

    design_result.initial_result = initial_result

    # initialize as empty list
    dict_of_timepoints_cond = {key: [] for key in
                               design_problem.petab_problem.condition_df
                                   .index.to_list()}

    # for each experimental conditions, save the time points at which we
    # consider a measurement for any candidate in experimental_list
    for candidate in design_problem.experiment_list:
        measurement_df = candidate['measurement_df']
        for ind, cond in enumerate(measurement_df.simulationConditionId):
            dict_of_timepoints_cond[cond].append(measurement_df.time[ind])

    # efficiency ?
    for cond in dict_of_timepoints_cond:
        dict_of_timepoints_cond[cond] = sorted(
            list(set(dict_of_timepoints_cond[cond])))

    # simulate all conditions forward once, save jacobian in a similar dict
    # as above
    deriv_dict = get_derivatives(design_problem,
                                 dict_of_timepoints_cond,
                                 x)

    for cand_ind, candidate in enumerate(design_problem.experiment_list):
        # new_FIM = old_FIM + A*A^T
        A = get_fim_addition(design_result,
                             candidate,
                             deriv_dict,
                             dict_of_timepoints_cond)

        if np.isnan(A).any():
            # TODO this case is not handled tight now
            print("SIMULATION FAILED")
            design_result.single_runs[index].append(
                get_design_result(design_problem=design_problem,
                                  candidate=candidate,
                                  x=x,
                                  hess=None))
        else:
            design_result.single_runs[index].append(
                get_design_result(design_problem=design_problem,
                                  candidate=candidate,
                                  x=x,
                                  hess=initial_result['hess'],
                                  hess_additional=A,
                                  initial_result=initial_result))

        design_result.single_runs[index][cand_ind][
            'hess_additional'] = A
        # design_result.single_runs[index][cand_ind]['fim_added'] = \
        #     initial_result['hess'] + fim_addition

    best_value, best_index = \
        design_result.get_best_conditions(key=design_problem.chosen_criteria,
                                          index=index)
    design_result.best_value.append(best_value)
    design_result.best_index.append(best_index)
    return design_result


def single_design_average(design_result: DesignResult):
    average_design_result = []
    for i in range(len(design_result.single_runs[0])):
        list_of_dicts = [yo[i] for yo in design_result.single_runs]

        average_design_result.append(get_average_result_dict(list_of_dicts))
    return average_design_result


def run_exp_design(design_problem: DesignProblem) -> DesignResult:
    """
    The main method for experimental design.

    Parameters
    ----------
    design_problem:
        the  problem formulation for experimental design

    Returns
    -------
    design_result:
        the result object which contains criteria values for each candidate
        condition which is to be tested
    """
    design_result = DesignResult(design_problem=design_problem)

    # if only one x is passed, convert it into a list of lists
    if not all(isinstance(elem, Iterable) for elem in
               design_problem.initial_x):
        design_problem.initial_x = [design_problem.initial_x]

    # create empty lists for the results of the single candidates for each
    # parameter set in 'design_problem.initial_x'
    design_result.single_runs = [[] for _ in
                                 range(len(design_problem.initial_x))]

    # compute the results for each candidate specified in
    # design_problem.experimental_list for each x in design_problem.initial_x
    # save result in a list in design_result.single_runs
    for i, x in enumerate(design_problem.initial_x):
        design_result = single_design_algo(design_result=design_result,
                                           x=x,
                                           index=i)
    # if multiple sets of parameters were passed, compute average values for
    # the criteria that will be used later when checking combinations
    if len(design_problem.initial_x) > 1:
        design_result.single_runs_average = single_design_average(
            design_result=design_result)
    return design_result
