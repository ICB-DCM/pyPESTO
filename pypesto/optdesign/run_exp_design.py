from .design_problem import DesignProblem
from .result import DesignResult
from .opt_design_helpers import get_design_result
from .change_dataframe import get_fim_addition, get_derivatives
import numpy as np


def single_design_algo(design_problem: DesignProblem,
                       design_result: DesignResult
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
    design_problem:
        problem formulation for an experimental design setting
    design_result:
        result object for an experimental design setting

    Returns
    -------
    design_result:
        the altered design_result
    """
    initial_result = get_design_result(design_problem=design_problem,
                                       candidate=None,
                                       x=design_problem.initial_x)
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
                                 dict_of_timepoints_cond)

    for cand_ind, candidate in enumerate(design_problem.experiment_list):
        fim_addition = get_fim_addition(design_problem,
                                        candidate,
                                        deriv_dict,
                                        dict_of_timepoints_cond)

        if np.isnan(fim_addition).any():
            design_result.single_runs.append(
                get_design_result(design_problem=design_problem,
                                  candidate=candidate,
                                  x=design_problem.initial_x,
                                  hess=None))
        else:
            design_result.single_runs.append(
                get_design_result(design_problem=design_problem,
                                  candidate=candidate,
                                  x=design_problem.initial_x,
                                  hess=initial_result['hess'] + fim_addition))

        design_result.single_runs[cand_ind]['fim_addition'] = fim_addition
        design_result.single_runs[cand_ind]['fim_added'] = \
            initial_result['hess'] + fim_addition

        if design_problem.profiles:
            raise NotImplementedError
            # plot_profile(result=result, problem=problem, obj=obj, index=0)

    design_result.best_value, design_result.best_index = \
        design_result.get_best_conditions(key=design_problem.chosen_criteria)
    return design_result


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

    design_result = single_design_algo(design_problem=design_problem,
                                       design_result=design_result)

    return design_result
