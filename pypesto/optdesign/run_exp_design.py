from .design_problem import DesignProblem
from .result import DesignResult
from .opt_design_helpers import update_pypesto_from_petab, get_design_result
from .change_dataframe import add_candidate_to_dfs, delete_candidate_from_dfs
from .optimize import optimization
from typing import List


# TODO keep track of changes to design_problem
def single_design_algo(design_problem: DesignProblem,
                       design_result: DesignResult
                       ) -> (DesignResult, List[float]):

    artificial_data = []
    runs_for_this_round = []

    for candidate in design_problem.problem_list:

        design_problem, written_values = add_candidate_to_dfs(design_problem=design_problem, candidate=candidate,
                                                              x=design_problem.result.optimize_result.as_list(['x'])[0][
                                                                  'x'])
        artificial_data.append(written_values)

        design_problem = update_pypesto_from_petab(design_problem)

        result = optimization(design_problem=design_problem)

        runs_for_this_round.append(
            get_design_result(design_problem=design_problem, candidate=candidate, fn=None, result=result))
        delete_candidate_from_dfs(design_problem=design_problem, candidate=candidate)

        if design_problem.profiles:
            raise NotImplementedError
            # plot_profile(result=result, problem=problem, obj=obj, index=0)

    design_result.single_runs.append(runs_for_this_round)
    return design_result, artificial_data


def run_exp_design(design_problem: DesignProblem) -> DesignResult:
    design_result = DesignResult(design_problem=design_problem)

    # loop for how many conditions we want to add in general
    # only works for one right now
    for run_index in range(design_problem.n_cond_to_add):
        design_result, artificial_data = single_design_algo(design_problem=design_problem, design_result=design_result)

        best_value, best_index = design_result.get_best_conditions(run=run_index)
    return design_result
