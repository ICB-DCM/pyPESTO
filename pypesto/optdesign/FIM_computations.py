import numpy as np
# from .result import DesignResult
from .opt_design_helpers import get_design_result
from .design_problem import DesignProblem
import amici
from typing import Iterable, List


# TODO rename this file to something like "FIM_computations.py" or similar

def get_fim_addition(design_problem: DesignProblem,
                     candidate: dict,
                     deriv_dict: dict,
                     dict_of_timepoints_cond: dict):
    """
    uses the previously computed parameter derivatives of the observables
    and computes the effect of a candidate on the FIM
    returns NOT the addition to the FIM but a matrix A such that
    new_FIM = old_FIM + A*A^T
    A can also be used to compute the new determinant without computing
    eigenvalues

    Parameters
    ----------
    design_problem: the optimal design problem
    candidate: dict with an id, measurement_df, (condition_df, observable_df)
    with candidate measurements that we want to check
    deriv_dict: a dictionary, contains for each experimental condition the
              parameter derivatives of the observables
    dict_of_timepoints_cond: a dictionary with the time points that we are
                             interested in for each condition

    Returns
    -------
    A: matrix A such that A*A^T is the total addition to the FIM
    """

    model = design_problem.problem.objective.amici_model

    A = np.nan * np.ones((len(design_problem.initial_x[0]),
                          len(candidate['measurement_df'])))
    for row_index in range(len(candidate['measurement_df'])):
        cond_name = candidate['measurement_df'].simulationConditionId[
            row_index]
        deriv_cond = deriv_dict[cond_name]

        # eg fixed noise parameters
        missing_params = len(model.getParameterIds()) - len(
            design_problem.initial_x[0])

        obs_position = model.getObservableIds().index(
            candidate['measurement_df'].observableId[row_index])
        time_position = dict_of_timepoints_cond[cond_name].index(
            candidate['measurement_df'].time[row_index])
        jac = deriv_cond[time_position][0:len(deriv_cond[0]) - missing_params,
                                        obs_position]

        # in design_problem.number_of_measurements one can specify how many
        # measurements should be taken, this effectively reduces the noise
        # value
        A[:, row_index] = jac * np.sqrt(
            design_problem.number_of_measurements) \
            / candidate['measurement_df'].noiseParameters[row_index]

        # single_addition = np.outer(jac, jac) / (
        #         candidate['measurement_df'].noiseParameters[
        #             row_index] ** 2)
        # total_addition = total_addition + (
        #         design_problem.number_of_measurements *
        #         single_addition)
    return A


def get_derivatives(design_problem: DesignProblem,
                    dict_of_timepoints_cond: dict,
                    x: Iterable) -> dict:
    """
    does a forward simulation for all experimental conditions and returns a
    dictionary of the parameter derivatives of observables for each condition

    Parameters
    ----------
    design_problem: the optimal design problem
    dict_of_timepoints_cond: a dictionary with the time points that we are
    interested in for each condition
    x: the single set of parameters that will be used
    Returns
    -------
    deriv_dict: dictionary of the parameter derivatives of observables for
    each condition
    """
    deriv_dict = {}
    model = design_problem.problem.objective.amici_model
    solver = model.getSolver()
    solver.setSensitivityOrder(amici.SensitivityOrder_first)
    model.setParameterScale(amici.ParameterScaling_log10)
    temp = x

    # these may be noise parameters for which we have explicit values
    missing_params = len(model.getParameterIds()) - len(temp)
    for i in range(missing_params):
        temp = np.append(temp, 0)
    model.setParameters(amici.DoubleVector(temp))

    for ind, condition in enumerate(
            design_problem.petab_problem.condition_df.index):
        all_fixed_params = []
        # set fixed parameters for condition
        for fixed_par in model.getFixedParameterIds():
            all_fixed_params.append(
                design_problem.petab_problem.condition_df[fixed_par][ind])
        model.setFixedParameters(
            amici.DoubleVector(np.array(all_fixed_params)))

        # set time points we are interested in
        model.setTimepoints(
            amici.DoubleVector(dict_of_timepoints_cond[condition]))

        rdata = amici.runAmiciSimulation(model, solver, None)
        vec = rdata['sy']

        deriv_dict[condition] = vec
    return deriv_dict


def get_combi_run_result(relevant_single_runs: List[dict],
                         initial_result: dict,
                         design_problem: DesignProblem,
                         combi: list,
                         x: Iterable) \
        -> dict:
    """
    computes the new criteria values etc in a dict after adding the new
    addition to the FIM

    Parameters
    ----------
    relevant_single_runs: the result of the candidates specified in the
    experiment list. 'relevant' because only the ones for this particular x
    initial_result: the result without adding any new measurements for this x
    design_problem: the problem formulation
    combi: a list of indices specifying which candidates from
    design_problem.experiment_list should be combined
    x: current set of parameters used

    Returns
    -------
    result: a dict with info about the combination specified in 'combi'
            saves in particular criteria values
    """
    # we will consider FIM_new = FIM_old + v*v^T + w*w^T + x*x^T + ...
    # where v,w,x are vectors and '*' is the outer product
    # one can write v*v^T + w*w^T + x*x^T + ... = M*M^T where M is the matrix
    # with v,w,x,... as columns
    # here we create this Matrix as 'total_hess_additional', and use it in
    # 'get_design_result' to compute the new FIM + criteria

    total_hess_additional = relevant_single_runs[combi[0]]['hess_additional']
    for index in combi[1:]:
        total_hess_additional = np.hstack((total_hess_additional,
                                           relevant_single_runs[index][
                                               'hess_additional']))

    result = get_design_result(
        design_problem=design_problem,
        candidate=combi,
        x=x,
        hess=initial_result['hess'],
        hess_additional=total_hess_additional)

    return result
