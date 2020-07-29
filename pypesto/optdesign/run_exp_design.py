from .design_problem import DesignProblem
from .result import DesignResult
from .opt_design_helpers import update_pypesto_from_petab, get_design_result
from .change_dataframe import add_candidate_to_dfs, delete_candidate_from_dfs
from .optimize import optimization


def single_design_algo(design_problem: DesignProblem,
                       design_result: DesignResult
                       ) -> DesignResult:
    """
    Algorithm to find the single best condition to be added.
    For all candidates in design_problem.problem_list measurements are
    simulated, added to the problem and a multistart optimization is run to
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
    artificial_data:
        the measurement values which where simulated
    """
    runs_for_this_round = []

    for candidate in design_problem.problem_list:

        design_problem = add_candidate_to_dfs(
            design_problem=design_problem, candidate=candidate,
            x=design_problem.result.optimize_result.as_list(['x'])[0]['x'])

        design_problem = update_pypesto_from_petab(design_problem)

        result = optimization(design_problem=design_problem)

        runs_for_this_round.append(
            get_design_result(design_problem=design_problem,
                              candidate=candidate, fn=None, result=result))
        delete_candidate_from_dfs(design_problem=design_problem,
                                  candidate=candidate)

        if design_problem.profiles:
            raise NotImplementedError
            # plot_profile(result=result, problem=problem, obj=obj, index=0)

    design_result.single_runs.append(runs_for_this_round)
    return design_result


def run_exp_design(design_problem: DesignProblem) -> DesignResult:
    """
    The main method for experimental design.
    Repeatedly searches for the single best condition to be added

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

    # loop for how many conditions we want to add in general
    # only works for one right now
    for run_index in range(design_problem.n_cond_to_add):
        design_result = single_design_algo(design_problem=design_problem,
                                           design_result=design_result)

        # TODO store best conditions for each round in a reasonable way
        best_value, best_index = design_result.get_best_conditions(
            run=run_index)
    return design_result
