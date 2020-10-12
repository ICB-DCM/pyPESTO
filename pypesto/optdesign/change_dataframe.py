import numpy as np
from .result import DesignResult
from .design_problem import DesignProblem
import amici
from typing import Iterable


def get_fim_addition(design_result: DesignResult,
                     candidate,
                     deriv_dict,
                     dict_of_timepoints_cond):
    """
    uses the previously computed parameter derivatives of the observables
    and computes the effect of a candidate on the FIM
    returns NOT the addition to the FIM but a matrix A such that
    new_FIM = old_FIM + A*A^T
    A can also be used to compute the new determinant without computing
    eigenvalues

    Parameters
    ----------
    design_result: the optimal design result
    candidate: measurement_df, (condition_df, observable_df) with candidate
               measurements that we want to check
    deriv_dict: a dictionary, contains for each experimental condition the
              parameter derivatives of the observables
    dict_of_timepoints_cond: a dictionary with the time points that we are
                             interested in for each condition

    Returns
    -------
    A: matrix A such that A*A^T is the total addition to the FIM
    """

    design_problem = design_result.design_problem
    A = np.nan * np.ones((len(design_problem.initial_x[0]),
                          len(candidate['measurement_df'])))
    for row_index in range(len(candidate['measurement_df'])):
        cond_name = candidate['measurement_df'].simulationConditionId[
            row_index]
        deriv_cond = deriv_dict[cond_name]

        # ie fixed noise parameters
        missing_params = len(design_problem.model.getParameterIds()) - len(
            design_problem.initial_x[0])

        obs_position = design_problem.model.getObservableIds().index(
            candidate['measurement_df'].observableId[row_index])
        time_position = dict_of_timepoints_cond[cond_name].index(
            candidate['measurement_df'].time[row_index])
        jac = deriv_cond[time_position][0:len(deriv_cond[0]) -
                                          missing_params, obs_position]

        A[:, row_index] = jac * np.sqrt(
            design_problem.number_of_measurements) \
                          / candidate['measurement_df'].noiseParameters[
                              row_index]

        # single_addition = np.outer(jac, jac) / (
        #         candidate['measurement_df'].noiseParameters[
        #             row_index] ** 2)
        # total_addition = total_addition + (
        #         design_problem.number_of_measurements *
        #         single_addition)
    # total_addition = np.matmul(A, A.transpose())
    return A


def get_derivatives(design_problem: DesignProblem,
                    dict_of_timepoints_cond: dict,
                    x: Iterable):
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

    model = design_problem.model
    solver = model.getSolver()
    solver.setSensitivityOrder(amici.SensitivityOrder_first)
    model.setParameterScale(amici.ParameterScaling_log10)
    temp = x

    # solver.setMaxSteps(1000000)
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
