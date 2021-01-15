import logging
from typing import Callable

from ..engine import Task
from ..problem import Problem
from .result import ProfilerResult
from .options import ProfileOptions
from .walk_along_profile import walk_along_profile
import pypesto.optimize

logger = logging.getLogger(__name__)


class ProfilerTask(Task):
    """A parameter likelihood profiling task."""

    def __init__(
            self,
            current_profile: ProfilerResult,
            problem: Problem,
            options: ProfileOptions,
            i_par: int,
            global_opt: float,
            optimizer: 'pypesto.optimize.Optimizer',
            create_next_guess: Callable,
    ):
        """
        Create the task object.

        Parameters
        ----------
        current_profile:
            The profile which should be computed
        problem:
            The problem to be solved.
        optimizer:
            The optimizer to be used along each profile.
        global_opt:
            log-posterior value of the global optimum
        options:
            Various options applied to the profile optimization.
        create_next_guess:
            Handle of the method which creates the next profile point proposal
        i_par:
            index for the current parameter
        """
        super().__init__()

        self.optimizer = optimizer
        self.problem = problem
        self.current_profile = current_profile
        self.global_opt = global_opt
        self.create_next_guess = create_next_guess
        self.i_par = i_par
        self.options = options

    def execute(self) -> 'pypesto.profile.ProfilerResult':
        logger.info(f"Executing task {self.i_par}.")
        # compute profile in descending and ascending direction
        for par_direction in [-1, 1]:
            # flip profile
            self.current_profile.flip_profile()

            # compute the current profile
            self.current_profile = walk_along_profile(
                current_profile=self.current_profile,
                problem=self.problem,
                par_direction=par_direction,
                optimizer=self.optimizer,
                options=self.options,
                create_next_guess=self.create_next_guess,
                global_opt=self.global_opt,
                i_par=self.i_par)

        # return the ProfilerResult and the index of the parameter profiled
        return {'profile': self.current_profile, 'index': self.i_par}
