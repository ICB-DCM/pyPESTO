import pandas as pd
from itertools import combinations
from .design_problem import DesignProblem
from .result import DesignResult
from .opt_design_helpers import update_pypesto_from_petab, get_design_result
from petab.C import CONDITION_ID, OBSERVABLE_ID
from .change_dataframe import add_candidate_to_dfs, delete_candidate_from_dfs
from .optimize import optimization


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
    for candidate in design_problem.experiment_list:
        design_problem = add_candidate_to_dfs(
            design_problem=design_problem, candidate=candidate,
            x=design_problem.initial_x)

        design_problem = update_pypesto_from_petab(design_problem)

        if design_problem.run_optimization:
            result = optimization(design_problem=design_problem)

            design_result.single_runs.append(
                get_design_result(design_problem=design_problem,
                                  candidate=candidate,
                                  x=result.optimize_result.as_list(
                                      ['x'])[0]['x']))
        else:
            design_result.single_runs.append(
                get_design_result(design_problem=design_problem,
                                  candidate=candidate,
                                  x=design_problem.initial_x))

        delete_candidate_from_dfs(design_problem=design_problem,
                                  candidate=candidate)

        if design_problem.profiles:
            raise NotImplementedError
            # plot_profile(result=result, problem=problem, obj=obj, index=0)

    design_result.best_value, design_result.best_index = \
        design_result.get_best_conditions(key=design_problem.chosen_criteria)
    return design_result


def do_combinatorics(design_problem: DesignProblem) -> DesignProblem:
    index_combinations = list(combinations(range(len(
        design_problem.experiment_list)), 2))
    new_experiment_list = []
    for indices in index_combinations:
        candidates = [design_problem.experiment_list[i] for i in indices]

        new_conditions = [dict['condition_df'] for dict in candidates
                          if 'condition_df' in dict]
        if all(v is None for v in new_conditions):
            new_conditions = None
        else:
            new_conditions = pd.concat(new_conditions)
            new_conditions = new_conditions.reset_index().drop_duplicates(
            ).set_index(CONDITION_ID)

        new_observables = [dict['observable_df'] for dict in candidates if
                           'observable_df' in dict]
        if all(v is None for v in new_observables):
            new_observables = None
        else:
            new_observables = pd.concat(new_observables)
            new_observables = new_observables.reset_index().drop_duplicates(
            ).set_index(OBSERVABLE_ID)

        new_measurements = [dict['measurement_df'] for dict in candidates]
        new_measurements = pd.concat(new_measurements, ignore_index=True)
        new_measurements = new_measurements.drop_duplicates()

        # TODO if condition_df, observable_df was not given, do not include
        #  it here
        new_dict = {'id': [dict['id'] for dict in candidates],
                    'condition_df': new_conditions,
                    'observable_df': new_observables,
                    'measurement_df': new_measurements}
        new_experiment_list.append(new_dict)

    design_problem.experiment_list = new_experiment_list
    return design_problem


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

    if design_problem.n_cond_to_add > 1:
        design_problem = do_combinatorics(design_problem=design_problem)

    design_result = single_design_algo(design_problem=design_problem,
                                       design_result=design_result)
    return design_result
