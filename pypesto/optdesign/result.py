from typing import List
from .design_problem import DesignProblem
from heapq import nlargest, nsmallest


class DesignResult(dict):

    def __init__(self,
                 design_problem: DesignProblem,
                 single_runs: List[dict] = None):

        super().__init__()

        if single_runs is None:
            single_runs = []
        self.design_problem = design_problem
        self.single_runs = single_runs
        self.best_value = None
        self.best_index = None

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def get_criteria_values(self, criteria: str):

        return [d[criteria] for d in self.single_runs]

    def get_best_conditions(self,
                            key: str = None,
                            n_best: int = 1,
                            maximize: bool = True) \
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

        Returns
        -------
        best_values, best_combination:
            the tuple consisting of two lists with the best values for the
            criteria and the the list of the best indices
        """

        if key is None:
            key = self.design_problem.chosen_criteria

        values = [d[key] for d in self.single_runs]

        if maximize:
            best_indices, best_values = map(list, zip(
                *nlargest(n_best, enumerate(values), key=lambda x: x[1])))
        else:
            best_indices, best_values = map(list, zip(
                *nsmallest(n_best, enumerate(values), key=lambda x: x[1])))

        best_combination = [self.single_runs[i]['candidate']['id'] for i
                            in best_indices]
        return best_values, best_combination

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
