from typing import List, Union
from .design_problem import DesignProblem
from heapq import nlargest, nsmallest
from .opt_design_helpers import get_design_result, combinations_gen
import numpy as np


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
        self.combi_runs = combi_runs
        self.initial_result = initial_result
        self.best_value = None
        self.best_index = None
        self.list_of_combinations = None

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def get_criteria_values(self, criteria: str, runs: str = None):
        result = None

        if runs is None or runs == 'single_runs':
            result = [d[criteria] for d in self.single_runs]
        elif runs == 'combi_runs':
            result = [d[criteria] for d in self.combi_runs[criteria]]
        else:
            print("Could not get criteria values. Specify if it should be "
                  "from 'single_runs' or 'combi_runs'")

        return result

    def get_best_conditions(self,
                            key: str = None,
                            n_best: int = 1,
                            maximize: bool = True,
                            runs: str = 'single_runs') \
            -> (List[float], List[int]):
        """
        returns a tuple of the n_best many best values and indices from
        'experiment_list' for the specified criteria
        if n_cond_to_add > 1,  best_combination will be a list with the best
        indices to combine

        Parameters
        ----------
        key: the name of the criteria to compare
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

        if key is None:
            key = self.design_problem.chosen_criteria
        if runs == 'single_runs':
            values = [d[key] for d in self.single_runs]
        elif runs == 'combi_runs':
            # self.combi_runs for each criteria are unsorted
            # temporarily save a sorted version
            sorted_combi_runs = {}
            for criteria in self.combi_runs:
                sorted_combi_runs[criteria] = sorted(self.combi_runs[criteria],
                                                     key=lambda d: d[criteria],
                                                     reverse=True)

            values = [best[key] for best in sorted_combi_runs[key]]
            indices = [best['candidate'] for best in sorted_combi_runs[key]]
        else:
            raise ValueError("can't find the specified runs to get "
                             "conditions from")
        # some criteria values may be None if the simulation failed etc
        # write -inf, +inf respectively for the code to work
        none_ind = np.where(np.array(values) == None)[0].tolist()

        if maximize:
            values = [float('-inf') if i in none_ind else value for i,
                      value in enumerate(values)]
            best_indices, best_values = map(list, zip(
                *nlargest(n_best, enumerate(values), key=lambda x: x[1])))
        else:
            values = [float('inf') if i in none_ind else value for i,
                      value in enumerate(values)]
            best_indices, best_values = map(list, zip(
                *nsmallest(n_best, enumerate(values), key=lambda x: x[1])))

        if runs == 'single_runs':
            best_combination = [self.single_runs[i]['candidate']['id'] for i
                                in best_indices]
        elif runs == 'combi_runs':
            best_combination = [indices[i] for i in best_indices]
        else:
            raise ValueError("can't find the specified runs to get "
                             "conditions from")
        return best_values, best_combination

    def get_combi_run_result(self,
                             list_of_combinations: list) \
            -> List[dict]:
        """
        computes the new criteria values etc in a dict after adding the new
        addition
        to the fim

        Parameters
        ----------
        list_of_combinations

        Returns
        -------

        """
        combi_runs = []
        for combi in list_of_combinations:
            new_hess = self.initial_result['hess']
            for index in combi:
                new_hess = new_hess + self.single_runs[index][
                    'fim_addition']
            new_result = get_design_result(
                design_problem=self.design_problem,
                candidate=combi,
                x=self.design_problem.initial_x,
                hess=new_hess)
            combi_runs.append(new_result)

        return combi_runs

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
        self.combi_runs = {key: [] for key in
                           self.design_problem.criteria_list}
        if self.design_problem.const_for_hess:
            for criteria in self.design_problem.criteria_list:
                self.combi_runs[criteria + '_modified'] = []

        # gets combinations of indices for entries in 'experiment_list'
        if isinstance(list_of_combinations, int):
            generator = combinations_gen(elements=range(len(
                self.design_problem.experiment_list)),
                length=list_of_combinations)
        else:
            generator = (element for element in list_of_combinations)

        # TODO change implementation of the modified version
        # include it in criteria_list !!!

        # keep track of the lowest value for each criteria in the list
        # initialize with -infinity
        criteria_min = {}
        for criteria in self.design_problem.criteria_list:
            criteria_min[criteria] = float("inf")
        if self.design_problem.const_for_hess:
            for criteria in self.design_problem.criteria_list:
                criteria_min[criteria + '_modified'] = float("inf")

        # for each combination to check see if it is in the best n many
        # if yes override saved entry
        for combi in generator:
            dict_result = \
                self.get_combi_run_result(list_of_combinations=[combi])[0]
            for criteria in self.design_problem.criteria_list:
                if len(self.combi_runs[criteria]) < \
                        self.design_problem.n_save_combi_result:
                    self.combi_runs[criteria].append(dict_result)

                    # update lowest
                    if dict_result[criteria] < criteria_min[criteria]:
                        criteria_min[criteria] = dict_result[criteria]

                elif dict_result[criteria] > criteria_min[criteria]:
                    # (hopefully) fast way of getting the index
                    to_replace = dict(
                        (d[criteria], dict(d, index=index)) for (index, d) in
                        enumerate(
                            self.combi_runs[criteria])).get(
                        criteria_min[criteria])['index']
                    self.combi_runs[criteria][to_replace] = dict_result

                    criteria_min[criteria] = np.array([saved[criteria]
                                                       for saved in
                                                       self.combi_runs[
                                                           criteria]]).min()

            # same but for modified criteria
            if self.design_problem.const_for_hess:

                for criteria in self.design_problem.criteria_list:
                    criteria_mod = criteria + '_modified'
                    if len(self.combi_runs[criteria_mod]) < \
                            self.design_problem.n_save_combi_result:
                        self.combi_runs[criteria_mod].append(dict_result)

                        # update lowest
                        if dict_result[criteria_mod] < criteria_min[
                                criteria_mod]:
                            criteria_min[criteria_mod] = dict_result[
                                    criteria_mod]

                    elif dict_result[criteria_mod] > criteria_min[
                            criteria_mod]:
                        # (hopefully) fast way of getting the index
                        to_replace = dict(
                            (d[criteria_mod], dict(d, index=index)) for (
                                index, d)
                            in enumerate(self.combi_runs[criteria_mod])).get(
                            criteria_min[criteria_mod])['index']
                        self.combi_runs[criteria_mod][to_replace] = \
                            dict_result
                        criteria_min[criteria_mod] = np.array(
                            [saved[criteria_mod]
                             for saved in
                             self.combi_runs[
                                 criteria_mod]]).min()
        return self

    """
    # TODO implement as property ?
    @property
    def best_value(self):
        if self._best_value is None:
            self._best_value = self.get_best_conditions(
                self.design_problem.chosen_criteria)[0]
        return self._best_value

    @property
    def best_index(self):
        if self._best_index is None:
            self._best_index = self.get_best_conditions(
                self.design_problem.chosen_criteria)[1]
        return self._best_index
    """
