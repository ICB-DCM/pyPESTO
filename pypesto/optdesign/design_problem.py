import petab
import amici
from typing import Union, Optional, Iterable, List
from ..problem import Problem
from ..result import Result
from ..optimize.optimizer import Optimizer, ScipyOptimizer


# should we specify Optimizer here?

class DesignProblem(dict):

    def __init__(self,
                 problem_list: list,
                 model: Union['amici.Model', 'amici.ModelPtr'],
                 problem: Problem,
                 result: Result,
                 petab_problem: petab.Problem,
                 x: Optional[Iterable[float]] = None,
                 n_optimize_runs: int = 10,
                 n_cond_to_add: int = 1,
                 criteria_list: List[str] = None,
                 chosen_criteria: str = 'det',
                 const_for_hess: float = 10 ** (-4),
                 profiles: bool = False,
                 number_of_measurements: int = 3):

        super().__init__()

        if criteria_list is None:
            criteria_list = ['det', 'trace', 'ratio', 'rank', 'eigmin', 'number_good_eigvals']
        self.problem_list = problem_list
        self.model = model
        self.problem = problem
        self.result = result
        self.petab_problem = petab_problem
        self.x = x
        self.n_optimize_runs = n_optimize_runs
        self.n_cond_to_add = n_cond_to_add
        self.criteria_list = criteria_list
        self.chosen_criteria = chosen_criteria
        self.const_for_hess = const_for_hess
        self.profiles = profiles
        self.number_of_measurements = number_of_measurements

        # sanity checks for lengths of df in problem_list
        if not self.problem_list:
            raise ValueError('you need to pass a nonempty list of candidates')
        for dict in self.problem_list:
            if len(dict['condition_df']) != len(self.petab_problem.condition_df.reset_index().columns):
                raise ValueError('condition dataframe in given candidates has wrong length')
            if dict['observable_df'] and len(dict['observable_df']) != len(
                    self.petab_problem.observable_df.reset_index().columns):
                raise ValueError('observable dataframe in given candidates has wrong length')
            if len(dict['measurement_df']) != len(self.petab_problem.measurement_df.columns):
                raise ValueError('measurement dataframe in given candidates has wrong length')

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
