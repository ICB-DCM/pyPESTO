"""Manage all components of a pyPESTO model selection problem."""

import warnings
from typing import Any, Optional

import petab_select
from petab_select import Model, Models

from .method import MethodCaller
from .model_problem import TYPE_POSTPROCESSOR, ModelProblem  # noqa: F401


class Problem:
    """Handles use of a model selection algorithm.

    Handles model selection. Usage involves initialisation with a model
    specifications file, and then calling the :meth:`select` method to perform
    model selection with a specified algorithm and criterion.

    Attributes
    ----------
    calibrated_models : petab_select.Models
        All calibrated models.
    newly_calibrated_models : petab_select.Models
        All models that were calibrated in the latest iteration of model
        selection.
    method_caller : pypesto.select.method.MethodCaller
        Used to run a single iteration of a model
        selection method.
    model_problem_options : dict[str, typing.Any]
        Keyword arguments, passed to the constructor of
        :class:`ModelProblem`.
    petab_select_problem : petab_select.Problem
        A PEtab Select problem.
    """

    # FIXME rename `best_model` to `selected_model` everywhere

    def __init__(
        self,
        petab_select_problem: petab_select.Problem,
        model_postprocessor: Optional[TYPE_POSTPROCESSOR] = None,
        model_problem_options: dict = None,
    ):
        self.petab_select_problem = petab_select_problem

        self.model_problem_options = {}
        if model_problem_options is not None:
            self.model_problem_options = model_problem_options
        # TODO deprecated
        if model_postprocessor is not None:
            warnings.warn(
                "Specifying `model_postprocessor` directly is deprecated. "
                "Please specify it with `model_problem_options`, e.g. "
                'model_problem_options={"postprocessor": ...}`.',
                DeprecationWarning,
                stacklevel=1,
            )
            self.model_problem_options["postprocessor"] = model_postprocessor

        self.set_state(
            calibrated_models=Models(),
            newly_calibrated_models=Models(),
        )

        # TODO default caller, based on petab_select.Problem
        self.method_caller = None

    def create_method_caller(self, **kwargs) -> MethodCaller:
        """Create a method caller.

        ``kwargs`` are passed to the :class:`MethodCaller` constructor.

        Returns
        -------
        A :class:`MethodCaller` instance.
        """
        kwargs = kwargs.copy()
        model_problem_options = self.model_problem_options | kwargs.pop(
            "model_problem_options", {}
        )

        return MethodCaller(
            petab_select_problem=self.petab_select_problem,
            calibrated_models=self.calibrated_models,
            model_problem_options=model_problem_options,
            **kwargs,
        )

    def set_state(
        self,
        calibrated_models: Models,
        newly_calibrated_models: Models,
    ) -> None:
        """Set the state of the problem.

        See :class:`Problem` attributes for argument documentation.
        """
        self.calibrated_models = calibrated_models
        self.newly_calibrated_models = newly_calibrated_models

    def update_with_newly_calibrated_models(
        self,
        newly_calibrated_models: Optional[Models] = None,
    ) -> None:
        """Update the state of the problem with newly calibrated models.

        Args:
            newly_calibrated_models:
                See attributes of :class:`Problem`.
        """
        self.newly_calibrated_models = newly_calibrated_models
        self.calibrated_models += self.newly_calibrated_models

    def handle_select_kwargs(
        self,
        kwargs: dict[str, Any],
    ):
        """Check keyword arguments to select calls."""
        if "newly_calibrated_models" in kwargs:
            raise ValueError(
                "Please supply `newly_calibrated_models` via "
                "`pypesto.select.Problem.set_state`."
            )
        if "calibrated_models" in kwargs:
            raise ValueError(
                "Please supply `calibrated_models` via "
                "`pypesto.select.Problem.set_state`."
            )

    def select(
        self,
        **kwargs,
    ) -> tuple[Model, Models]:
        """Run a single iteration of a model selection algorithm.

        The result is the selected model for the current run, independent of
        previous selected models.

        ``kwargs`` are passed to the :class:`MethodCaller` constructor.

        Returns
        -------
        A 2-tuple, with the following values from this iteration:

           1. the best model; and
           2. all models.
        """
        self.handle_select_kwargs(kwargs)
        method_caller = self.create_method_caller(**kwargs)
        newly_calibrated_models = method_caller()

        self.update_with_newly_calibrated_models(
            newly_calibrated_models=newly_calibrated_models,
        )

        best_model = petab_select.ui.get_best(
            problem=self.petab_select_problem,
            models=self.newly_calibrated_models,
            criterion=method_caller.criterion,
        )

        return best_model, newly_calibrated_models

    def select_to_completion(
        self,
        **kwargs,
    ) -> Models:
        """Perform model selection until the method terminates.

        ``kwargs`` are passed to the :class:`MethodCaller` constructor.

        Returns
        -------
        All models.
        """
        calibrated_models = Models(problem=self.petab_select_problem)
        self.handle_select_kwargs(kwargs)
        method_caller = self.create_method_caller(**kwargs)

        while True:
            try:
                iteration_calibrated_models = method_caller()
                self.update_with_newly_calibrated_models(
                    newly_calibrated_models=iteration_calibrated_models,
                )
                calibrated_models += iteration_calibrated_models
            except StopIteration:
                break

        return calibrated_models

    # TODO method that automatically generates initial models, for a specific
    # number of starts. TODO parallelise?
    def multistart_select(
        self,
        predecessor_models: Models = None,
        **kwargs,
    ) -> tuple[Model, list[Model]]:
        """Run an algorithm multiple times, with different predecessor models.

        Note that the same method caller is currently shared between all calls.
        This may change when parallelization is implemented, but for now
        ensures that the same model isn't calibrated twice.
        Could also be managed by sharing the same "calibrated_models" object
        (but then the same model could be repeatedly calibrated, if the
        calibrations start before any have stopped).

        ``kwargs`` are passed to the :class:`MethodCaller` constructor.

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
        self.handle_select_kwargs(kwargs)
        model_lists = []

        for predecessor_model in predecessor_models:
            method_caller = self.create_method_caller(
                **(kwargs | {"predecessor_model": predecessor_model})
            )
            models = method_caller()
            self.calibrated_models += models

            model_lists.append(models)
            method_caller.candidate_space.reset()

        best_model = petab_select.ui.get_best(
            problem=method_caller.petab_select_problem,
            models=[model for models in model_lists for model in models],
            criterion=method_caller.criterion,
        )

        return best_model, model_lists
