from typing import List, Union, Iterable
from .design_problem import DesignProblem
from heapq import nlargest, nsmallest, heappush, heappushpop
from .opt_design_helpers import get_design_result, combinations_gen, \
    get_average_result_dict, divide_dict, add_to_dict
import numpy as np
from .change_dataframe import get_combi_run_result

from functools import partial
# from .task import combi_task
from ..engine import MultiThreadEngine, SingleCoreEngine
from multiprocessing import Manager, Pool


class CombiRes:
    """
    Use this class to save individual results
    We need it to perform heap-operations while also saving an entire
    dictionary at the same time, hence the special "<", ">" operators.


    Was mostly needed in case of parallelization, so multiple processes can
    heappush into a shared object. However this parallelization is not fully
    implemented right now.
    """

    def __init__(self, dict, criteria):
        self.dict = dict
        self.value = dict[criteria]
        self.criteria = criteria

    def __lt__(self, other):  # To override > operator
        return self.value < other.value

    def __gt__(self, other):  # To override < operator
        return self.value > other.value

    def get_dict(self):
        return self.dict

    def get_value(self, criteria=None):
        if criteria is None:
            criteria = self.criteria
        return self.dict[criteria]


class DesignResult(dict):

    def __init__(self,
                 design_problem: DesignProblem,
                 single_runs: List[dict] = None,
                 combi_runs: List[dict] = None,
                 initial_result: List[dict] = None):

        super().__init__()

        if single_runs is None:
            single_runs = []
        self.design_problem = design_problem
        self.single_runs = single_runs
        self.single_runs_average = None
        self.combi_runs = combi_runs
        self.initial_result = initial_result
        self.list_of_combinations = None

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def get_criteria_values(self,
                            criteria: str,
                            runs: str = 'single_runs',
                            x_index: Union[int, str] = None) \
            -> List[float]:
        """
        returns a list of the criteria values for the specified criteria
        For runs = 'single_runs' returned list is not sorted by value but by
        the order of design_problem.experiment_list
        For runs = 'combi_runs' the list is sorted by the criteria value in
        descending order

        Parameters
        ----------
        criteria: the criteria from self.design_problem.criteria_list for
                  which you want the values
        runs: either 'single_runs' for the values of candidates specified in
              'design_problem.experiment_list' or 'combi_runs' for the best
              values of combinations of candidates
        x_index: important if runs='single_runs'. if design_problem.initial_x
                 is a list, use an integer to get the values
                 for this parameter set. Use 'average' to get the averaged
                 value across all parameter sets
        """
        result = None
        if x_index is None:
            if len(self.design_problem.initial_x) > 1:
                x_index = 'average'
            else:
                x_index = 0
        if runs == 'single_runs':
            if isinstance(x_index, int):
                result = [d[criteria] for d in self.single_runs[x_index]]
            elif x_index == 'average':
                result = [d[criteria] for d in self.single_runs_average]
            else:
                print("Could not get criteria values. Specify x_index as an "
                      "integer for the parameter values or 'average'")
        elif runs == 'combi_runs':
            result = [d[criteria] for d in self.combi_runs[criteria]]
        else:
            print("Could not get criteria values. Specify if it should be "
                  "from 'single_runs' or 'combi_runs'")

        return result

    def get_best_conditions(self,
                            criteria: str,
                            index: Union[int, str] = None,
                            n_best: int = 1,
                            maximize: bool = True,
                            runs: str = 'single_runs') \
            -> (List[float], List[int]):
        """
        returns a tuple of the n_best many best values and indices from
        'experiment_list' for the specified criteria
        if n_cond_to_add > 1,  the indices will consist of lists with the best
        indices to combine

        Parameters
        ----------
        index: specifies for which of the parameter sets the best criteria
               should be taken, either an integer or 'average' for the
               average result
        criteria: the name of the criteria to compare
        n_best: the first n_best best will be chosen
        maximize: if the criteria is to be maximized or minimized
        runs: specifies if the results for the single candidates or
        combinations of these should be evaluated

        Returns
        -------
        best_values, best_combination:
            the tuple consisting of two lists with the best values for the
            criteria and the the list of the best indices
        """

        # defaults to the average result in the case of multiple parameter sets
        if index is None:
            if len(self.design_problem.initial_x) > 1:
                index = 'average'
            else:
                index = 0
        if runs == 'single_runs':
            if isinstance(index, int):
                values = [d[criteria] for d in self.single_runs[index]]
            elif index == 'average':
                values = [d[criteria] for d in self.single_runs_average]
            else:
                print("Could not get criteria values. Specify x_index as an "
                      "integer for the parameter values or 'average'")

        elif runs == 'combi_runs':
            # self.combi_runs for each criteria are unsorted
            # temporarily save a sorted version
            sorted_combi_runs = {}
            for crit in self.combi_runs:
                sorted_combi_runs[crit] = sorted(self.combi_runs[crit],
                                                 key=lambda d: d[crit],
                                                 reverse=True)

            values = [best[criteria] for best in sorted_combi_runs[criteria]]
            indices = [best['candidate'] for best in sorted_combi_runs[
                criteria]]
        else:
            raise ValueError("can't find the specified runs to get "
                             "conditions from")
        # some criteria values may be None if the simulation failed etc
        # write -inf, +inf respectively for the code to work

        # note '== None' and 'is None' behave differently here
        # TODO change this so flake8 won't complain
        none_ind = np.where(np.array(values) == None)[0].tolist()

        # all currently used criteria are to be maximized
        # just in case there is also a routine to minimize
        if maximize:
            values = [float('-inf') if i in none_ind else value
                      for i, value in enumerate(values)]
            best_indices, best_values = map(list, zip(
                *nlargest(n_best, enumerate(values), key=lambda x: x[1])))
        else:
            values = [float('inf') if i in none_ind else value
                      for i, value in enumerate(values)]
            best_indices, best_values = map(list, zip(
                *nsmallest(n_best, enumerate(values), key=lambda x: x[1])))

        if runs == 'single_runs':
            best_combination = [self.single_runs[0][i]['candidate']['id'] for i
                                in best_indices]
        elif runs == 'combi_runs':
            best_combination = [indices[i] for i in best_indices]
        else:
            raise ValueError("can't find the specified runs to get "
                             "conditions from")
        return best_values, best_combination

    # this function was outsourced to "change_dataframe.py" so it it not a
    # method of the DesignResult anymore, so it can be called in case of
    # parallelization with one shared DesignResult

    # def get_combi_run_result(self,
    #                          list_of_combinations: list,
    #                          x: Iterable,
    #                          x_index: int,
    #                          criteria_min) \
    #         -> List[dict]:
    #     """
    #     computes the new criteria values etc in a dict after adding the new
    #     addition to the fim
    #
    #     Parameters
    #     ----------
    #     list_of_combinations:
    #     criteria_min: dict which has all criteria as keys. for each criteria
    #     save the lowest value of the n_save_combi_result that we save
    #     x: current set of parameters used
    #     x_index: index of x in design_problem.initial_x
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     combi_runs = []
    #     # shouldn't this list always have only a single element?
    #     # eg.[[0,1,4]] when adding combis of three measurements?
    #     for combi in list_of_combinations:
    #         total_hess_additional = np.zeros(self.single_runs[0][0][
    #                                              'hess_additional'].shape)
    #         # new_hess = self.initial_result['hess']
    #         for index in combi:
    #             total_hess_additional = np.hstack((total_hess_additional,
    #                                                self.single_runs[x_index][
    #                                                    index][
    #                                                    'hess_additional']))
    #
    #             # new_hess = new_hess + self.single_runs[x_index][index][
    #             #     'hess_additional']
    #
    #         new_result = get_design_result(
    #             design_problem=self.design_problem,
    #             candidate=combi,
    #             x=x,
    #             hess=self.initial_result[x_index]['hess'],
    #             hess_additional=total_hess_additional,
    #             initial_result=self.initial_result[x_index],
    #             combi_runs=self.combi_runs,
    #             skip=True,
    #             criteria_min=criteria_min)
    #         combi_runs.append(new_result)
    #
    #     return combi_runs

    def get_best_combi_index_pairs(self,
                                   criteria: str):
        """
        returns a list where each entry is a list of indices. Each list of
        indices describes a tested combination
        """
        list_of_combinations = [entry['candidate'] for entry in
                                self['combi_runs'][criteria]]
        return list_of_combinations

    def get_best_combi_index(self,
                             criteria: str):
        """
        similar to 'get_best_combi_index_pairs' but doesn't return a multiple
        lists of combinations but a list for all first entries, a list for
        all second entries etc.
        """
        list_of_combinations = self.get_best_combi_index_pairs(criteria)

        list_of_indices = [[combi[i] for combi in list_of_combinations] for
                           i in range(len(list_of_combinations[0]))]
        return list_of_indices

    def get_average_design_result(self, combi):

        average_design_result = get_combi_run_result(
            relevant_single_runs=self.single_runs[
                self.good_initial_x_indices[0]],
            combi=combi,
            x=self.design_problem.initial_x[
                self.good_initial_x_indices[0]],
            initial_result=self.initial_result[
                self.good_initial_x_indices[0]],
            design_problem=self.design_problem)

        for x_index in self.good_initial_x_indices[1:]:
            add = get_combi_run_result(
                relevant_single_runs=self.single_runs[x_index],
                combi=combi,
                x=self.design_problem.initial_x[x_index],
                initial_result=self.initial_result[x_index],
                design_problem=self.design_problem)
            average_design_result = add_to_dict(average_design_result, add)

        average_design_result = divide_dict(average_design_result, len(
            self.good_initial_x_indices))
        return average_design_result

    def check_combinations(self,
                           list_of_combinations: Union[List[List[int]], int]):
        """
        main routine for checking and saving the criteria values for
        combinations of candidates which where previously computed the
        'run_exp_design'
        results are saved in self.combi_runs

        Parameters
        ----------
        list_of_combinations:
            can be a list where each entry is a list of indices describing
            which entries in self.design_problem.experiment_list should be
            paired
            can be an integer n, then the whole combinatorial spaces of n
            combinations will be checked

        Returns
        -------

        """

        # set up combi_runs as dictionary with empty list as entry for
        # each criteria
        combi_runs = {key: [] for key in
                      self.design_problem.non_modified_criteria}
        if self.design_problem.modified_criteria:
            for crit in self.design_problem.modified_criteria:
                combi_runs[crit + '_modified'] = []

        # gets combinations of indices for entries in 'experiment_list'
        if isinstance(list_of_combinations, int):
            generator = combinations_gen(elements=range(len(
                self.design_problem.experiment_list)),
                length=list_of_combinations)
        else:
            generator = (element for element in list_of_combinations)

        # func = partial(combi_task, self)
        #
        # crit_min = Manager().dict()
        # combi_runs = Manager().dict()
        #
        # for crit in self.design_problem.non_modified_criteria:
        #     crit_min[crit] = float("inf")
        # if self.design_problem.modified_criteria:
        #     for crit in self.design_problem.modified_criteria:
        #         crit_min[crit + '_modified'] = float("inf")
        #
        # for key in self.design_problem.non_modified_criteria:
        #     combi_runs[key] = Manager().list()
        # if self.design_problem.modified_criteria:
        #     for crit in self.design_problem.modified_criteria:
        #         combi_runs[crit + '_modified'] = Manager().list()
        #
        # p = Pool(None, combi_task_init, [crit_min, combi_runs])
        #
        # p.imap(func, generator, chunksize=100)
        # p.close()
        # p.join()
        # print(combi_runs)
        # print(crit_min)
        # self.combi_runs = combi_runs

        for combi in generator:

            average_design_result = self.get_average_design_result(combi)

            # for each combination check if it is in the best
            # n_save_combi_result many. If yes override saved entry via
            # heappush
            # (implicitly assume all criteria need to be maximized)
            for crit in self.design_problem.non_modified_criteria:
                if len(combi_runs[crit]) < \
                        self.design_problem.n_save_combi_result:
                    heappush(combi_runs[crit], CombiRes(
                        average_design_result, crit))

                elif average_design_result[crit] > combi_runs[crit][0].value:
                    heappushpop(combi_runs[crit],
                                CombiRes(average_design_result, crit))

            # same but for modified criteria
            for crit in self.design_problem.modified_criteria:
                crit_mod = crit + '_modified'
                if len(combi_runs[crit_mod]) < \
                        self.design_problem.n_save_combi_result:
                    heappush(combi_runs[crit_mod], CombiRes(
                        average_design_result, crit_mod))

                elif average_design_result[crit_mod] > \
                        combi_runs[crit_mod][0].value:
                    heappushpop(combi_runs[crit_mod],
                                CombiRes(average_design_result, crit_mod))

        # change the class back to a list of dicts, the 'normal' format
        self.combi_runs = {}
        for crit in combi_runs:
            combi_runs[crit] = sorted(combi_runs[crit], reverse=True)
            self.combi_runs[crit] = [x.dict for x in combi_runs[crit]]
        return self
