from ..engine import Task
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
        from .run_exp_design import single_design_algo

        single_result = single_design_algo(design_result=self.design_result,
                                           x=self.x,
                                           index=self.index)

        return single_result
