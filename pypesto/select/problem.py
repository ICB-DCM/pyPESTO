"""Manage all components of a pyPESTO model selection problem."""
from typing import Iterable, List, Optional

import petab_select
from petab_select import Model

from ..C import TYPE_POSTPROCESSOR
from .method import MethodCaller


class Problem:
    """Handles use of a model selection algorithm.

    Handles model selection. Usage involves initialisation with a model
    specifications file, and then calling the `select()` method to perform
    model selection with a specified algorithm and criterion.

    Attributes
    ----------
    history:
        Storage for all calibrated models. A dictionary, where keys are
        model hashes, and values are `petab_select.Model` objects.
    method_caller:
        A `MethodCaller`, used to run a single iteration of a model
        selection method.
    model_postprocessor:
        A method that is applied to each model after calibration.
    petab_select_problem:
        A PEtab Select problem.
    """

    def __init__(
        self,
        petab_select_problem: petab_select.Problem,
        model_postprocessor: Optional[TYPE_POSTPROCESSOR] = None,
    ):
        self.petab_select_problem = petab_select_problem
        self.model_postprocessor = model_postprocessor
        self.history = {}

        # Dictionary of method names as keys, with a dictionary as the values.
        # In the dictionary, keys will be modelId, criterion value
        # TODO unused, might be useful for debugging
        self.results = {}

        # TODO default caller, based on petab_select.Problem
        self.method_caller = None

    def create_method_caller(self, *args, **kwargs) -> MethodCaller:
        """Create a method caller.

        `args` and `kwargs` are passed to the `MethodCaller` constructor.

        Returns
        -------
        A `MethodCaller` instance.
        """
        return MethodCaller(
            petab_select_problem=self.petab_select_problem,
            *args,
            history=self.history,
            model_postprocessor=self.model_postprocessor,
            **kwargs,
        )

    def select(
        self,
        *args,
        **kwargs,
    ):
        """Run a single iteration of a model selection algorithm.

        The result is the selected model for the current run, independent of
        previous selected models.

        `args` and `kwargs` are passed to the `MethodCaller` constructor.

        Returns
        -------
        A 3-tuple, with the following values:

           1. the best model;
           2. all candidate models in this iteration, as a `dict` with
              model hashes as keys and models as values; and
           3. all candidate models from all iterations, as a `dict` with
              model hashes as keys and models as values.
        """
        # TODO move some options to PEtab Select? e.g.:
        # - startpoint_latest_mle
        # - select_first_improvement

        # TODO handle bidirectional
        method_caller = self.create_method_caller(*args, **kwargs)
        best_model, local_history = method_caller()
        self.history.update(method_caller.history)

        # TODO: Reconsider return value. `result` could be stored in attribute,
        # then no values need to be returned, and users can request values
        # manually.
        # return result, self.selection_history
        return best_model, local_history, self.history

    def select_to_completion(
        self,
        *args,
        **kwargs,
    ) -> List[Model]:
        """Run an algorithm until an exception `StopIteration` is raised.

        `args` and `kwargs` are passed to the `MethodCaller` constructor.

        An exception `StopIteration` is raised by
        `pypesto.select.method.MethodCaller.__call__` when no candidate models
        are found.

        Returns
        -------
        The best models (the best model at each iteration).
        """
        best_models = []
        method_caller = self.create_method_caller(*args, **kwargs)

        intermediate_kwargs = {}
        while True:
            # TODO currently uses the best model so far, not the best model
            #      from the previous iteration. Make this a possibility?
            if best_models:
                # TODO string literal
                intermediate_kwargs["predecessor_model"] = best_models[-1]
            try:
                best_model, _ = method_caller(**intermediate_kwargs)
                if best_model is not None:
                    best_models.append(best_model)
                self.history.update(method_caller.history)
            except StopIteration:
                break

        return best_models

    # TODO method that automatically generates initial models, for a specific
    # number of starts. TODO parallelise?
    def multistart_select(
        self,
        *args,
        predecessor_models: Iterable[Model] = None,
        **kwargs,
    ):
        """Run an algorithm multiple times, with different predecessor models.

        Note that the same method caller is currently shared between all calls.
        This may change when parallelization is implemented, but for now
        ensures that the same model isn't calibrated twice.
        Could also be managed by sharing the same "history" object (but then
        the same model could be repeatedly calibrated, if the calibrations
        start before any have stopped).

        `args` and `kwargs` are passed to the `MethodCaller` constructor.

        Parameters
        ----------
        predecessor_models:
            The models that will be used as initial models. One "model
            selection iteration" will be run for each predecessor model.

        Returns
        -------
        A 2-tuple, with the following values:

           1. the best model; and
           2. the best models (the best model at each iteration).
        """
        best_models = []
        method_caller = self.create_method_caller(*args, **kwargs)
        for predecessor_model in predecessor_models:
            best_model, _ = method_caller(predecessor_model=predecessor_model)
            if best_model is not None:
                best_models.append(best_model)
            self.history.update(method_caller.history)

        best_model = petab_select.ui.best(
            problem=method_caller.petab_select_problem,
            models=best_models,
            criterion=method_caller.criterion,
        )

        return best_model, best_models
