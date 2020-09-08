import numpy as np
from .design_problem import DesignProblem
import amici


def get_fim_addition(design_problem: DesignProblem,
                     candidate,
                     deriv_dict,
                     dict_of_timepoints_cond):
    """
    uses the previously computed parameter derivatives of the observables
    and computes the effect of a candidate on the FIM
    returns the addition to the FIM

    Parameters
    ----------
    design_problem: the optimal design problem
    candidate: measurement_df, (condition_df, observable_df) with candidate
               measurements that we want to check
    deriv_dict: a dictionary, contains for each experimental condition the
              parameter derivatives of the observables
    dict_of_timepoints_cond: a dictionary with the time points that we are
                             interested in for each condition

    Returns
    -------
    total_addition: addition to the FIM
    """
    total_addition = 0
    for row_index in range(len(candidate['measurement_df'])):
        cond_name = candidate['measurement_df'].simulationConditionId[
            row_index]
        deriv_cond = deriv_dict[cond_name]

        # ie fixed noise parameters
        missing_params = len(design_problem.model.getParameterIds()) - len(
            design_problem.initial_x)

        obs_position = design_problem.model.getObservableIds().index(
            candidate['measurement_df'].observableId[row_index])
        time_position = dict_of_timepoints_cond[cond_name].index(
            candidate['measurement_df'].time[row_index])
        jac = deriv_cond[time_position][0:len(deriv_cond[0])
                                        - missing_params, obs_position]
        single_addition = np.outer(jac, jac) / (
                candidate['measurement_df'].noiseParameters[
                    row_index] ** 2)
        total_addition = total_addition + (
                design_problem.number_of_measurements *
                single_addition)

    return total_addition


def get_derivatives(design_problem: DesignProblem,
                    dict_of_timepoints_cond: dict):
    """
    does a forward simulation for all experimental conditions and returns a
    dictionary of the parameter derivatives of observables for each condition

    Parameters
    ----------
    design_problem: the optimal design problem
    dict_of_timepoints_cond: a dictionary with the time points that we are
    interested in for each condition

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
    temp = design_problem.initial_x

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
