import numpy as np
import logging

from ..engine import Task
# from ..objective import HistoryOptions
# from ..problem import Problem
# from .run_exp_design import single_design_algo
from .change_dataframe import get_combi_run_result
from .opt_design_helpers import get_average_result_dict, add_to_dict, \
    divide_dict
from multiprocessing import current_process
from typing import Iterable


class ExpDesignSingleTask(Task):
    """
    A ExpDesign task, which checks the the results for single candidates as
    specified in design_problem.experiment_list for one particular set of
    parameters, x.
    """

    def __init__(
            self,
            design_result,
            x: Iterable,
            index: int):
        super().__init__()

        self.design_result = design_result
        # TODO find a solution which doesn't simply override the problematic
        #  part
        # model as a SWIG object can't be pickled, so delete it
        self.design_result.design_problem.model = None
        self.x = x
        self.index = index

    def execute(self):
        # logger.info(f"Executing task {self.id}.")
        from .run_exp_design import single_design_algo

        single_result = single_design_algo(design_result=self.design_result,
                                           x=self.x,
                                           index=self.index)

        return single_result


# tries for parallelization:

# class ExpDesignCombiTask(Task):
#
#     def __init__(
#             self,
#             n_saved_eigmin,
#             relevant_single_runs,
#             combi,
#             x,
#             criteria_min,
#             initial_result,
#             design_problem
#     ):
#         super().__init__()
#
#         self.n_saved_eigmin = n_saved_eigmin
#         self.relevant_single_runs = relevant_single_runs
#         self.combi = combi
#         self.x = x
#         self.criteria_min = criteria_min
#         self.initial_result = initial_result
#         self.design_problem = design_problem
#
#     def execute(self) -> dict:
#         # logger.info(f"Executing task {self.id}.")
#
#         combi_result = get_combi_run_result(
#             n_saved_eigmin=self.n_saved_eigmin,
#             relevant_single_runs=self.relevant_single_runs,
#             initial_result=self.initial_result,
#             design_problem=self.design_problem,
#             combi=self.combi,
#             x=self.x,
#             criteria_min=self.criteria_min)[0]
#
#         return combi_result


# def combi_task(design_result,
#                combi: np.ndarray):
#     # print("starting: ", current_process().name)
#     # dict_list = []
#     #
#     # # only use the indices where no simulation failed
#     # for x_index in design_result.good_initial_x_indices:
#     #     dict_result = \
#     #         get_combi_run_result(
#     #             relevant_single_runs=design_result.single_runs[x_index],
#     #             combi=combi,
#     #             x=design_result.design_problem.initial_x[x_index],
#     #             initial_result=design_result.initial_result[x_index],
#     #             design_problem=design_result.design_problem)[0]
#     #     dict_list.append(dict_result)
#     #
#     # average_design_result = get_average_result_dict(dict_list)
#
#
#     average_design_result = get_combi_run_result(
#         relevant_single_runs=design_result.single_runs[
#             design_result.good_initial_x_indices[0]],
#         combi=combi,
#         x=design_result.design_problem.initial_x[
#             design_result.good_initial_x_indices[0]],
#         initial_result=design_result.initial_result[
#             design_result.good_initial_x_indices[0]],
#         design_problem=design_result.design_problem)
#
#     for x_index in design_result.good_initial_x_indices[1:]:
#         add = get_combi_run_result(
#             relevant_single_runs=design_result.single_runs[x_index],
#             combi=combi,
#             x=design_result.design_problem.initial_x[x_index],
#             initial_result=design_result.initial_result[x_index],
#             design_problem=design_result.design_problem)
#         average_design_result = add_to_dict(average_design_result, add)
#
#     average_design_result = divide_dict(average_design_result, len(
#         design_result.good_initial_x_indices))
#
#     if len(design_result.design_problem.initial_x) > 1:
#         average_design_result['x'] = 'average'
#         average_design_result['hess'] = 'average'
#         average_design_result['eigvals'] = 'average'
#
#     return average_design_result
