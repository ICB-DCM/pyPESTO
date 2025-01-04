"""Functionality related to using a PEtab Select model selection method."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
import petab_select
from petab_select import (
    CANDIDATE_SPACE,
    MODELS,
    PREDECESSOR_MODEL,
    UNCALIBRATED_MODELS,
    VIRTUAL_INITIAL_MODEL,
    CandidateSpace,
    Criterion,
    Method,
    Model,
    Models,
)

from ..problem import Problem
from .model_problem import TYPE_POSTPROCESSOR, ModelProblem


class MethodSignalProceed(str, Enum):
    """Indicators for how a model selection method should proceed."""

    # TODO move to PEtab Select?
    STOP = "stop"
    CONTINUE = "continue"


@dataclass
class MethodSignal:
    """The state of a model selection method after a single model calibration.

    Attributes
    ----------
    accept:
        Whether to accept the model.
    proceed:
        How the method should proceed.
    """

    accept: bool
    # TODO change to bool?
    proceed: MethodSignalProceed


class MethodLogger:
    """Log results from a model selection method.

    Attributes
    ----------
    column_width:
        The width of columns when logging.
    column_sep:
        The substring used to separate column values when logging.
    level:
        The logging level.
    logger:
        A logger from the :mod:`logging` module.
    """

    column_width: int = 12
    column_sep: str = " | "

    def __init__(self, level: str = "info"):
        self.logger = logging.getLogger(__name__)
        self.level = level

    def log(self, message, level: str = None) -> None:
        """Log a message.

        Parameters
        ----------
        message:
            The message.
        level:
            The logging level. Defaults to the value defined in the
            constructor.
        """
        if level is None:
            level = self.level
        getattr(self.logger, level)(message)

    def new_selection(self) -> None:
        """Start logging a new model selection."""
        padding = 20
        self.log("-" * padding + "New Selection" + "-" * padding)
        columns = {
            "Predecessor model subspace:ID": "model0",
            "Model subspace:ID": "model",
            "Criterion ID": "crit",
            "Predecessor model criterion": "model0_crit",
            "Model criterion": "model_crit",
            "Criterion difference": "crit_diff",
            "Accept": "accept",
        }
        columns = {
            k: v.ljust(self.column_width)[: self.column_width]
            for k, v in columns.items()
        }
        self.log(self.column_sep.join(columns.values()))

    def new_result(
        self,
        accept,
        criterion,
        model,
        predecessor_model,
        max_id_length: str = 12,
        precision: int = 3,
    ) -> None:
        """Log a model calibration result.

        Parameters
        ----------
        accept:
            Whether the model is accepted.
        criterion:
            The criterion type.
        max_id_length:
            Model and predecessor model IDs are truncated to this length in the
            logged message.
        model:
            The calibrated model.
        predecessor_model:
            The predecessor model.
        precision:
            The number of decimal places to log.
        """
        model_criterion = model.get_criterion(criterion)

        def get_model_id(model: Model) -> str:
            """Get a model ID for logging.

            Parameters
            ----------
            model:
                The model.

            Returns
            -------
            The ID.
            """
            model_subspace_id = model.model_subspace_id or ""
            original_model_id = model.model_id or model.get_hash()
            model_id = model_subspace_id + ":" + original_model_id
            return model_id

        def float_to_str(value: float, precision: int = 3) -> str:
            return f"{value:.{precision}e}"

        if isinstance(predecessor_model, Model):
            predecessor_model_id = get_model_id(predecessor_model)
            predecessor_model_criterion = predecessor_model.get_criterion(
                criterion
            )
            criterion_difference = float_to_str(
                model_criterion - predecessor_model_criterion
            )
            predecessor_model_criterion = float_to_str(
                predecessor_model_criterion
            )
        else:
            criterion_difference = None
            predecessor_model_criterion = None
            predecessor_model_id = predecessor_model

        model_criterion = float_to_str(model_criterion)

        message_parts = [
            predecessor_model_id,
            get_model_id(model),
            criterion.value,
            predecessor_model_criterion,
            model_criterion,
            criterion_difference,
            accept,
        ]
        message = self.column_sep.join(
            [
                str(v).ljust(self.column_width)[: self.column_width]
                for v in message_parts
            ]
        )
        self.log(message)


class MethodCaller:
    """Handle calls to PEtab Select model selection methods.

    Attributes
    ----------
    petab_select_problem:
        The PEtab Select problem.
    candidate_space:
        A :class:`petab_select.CandidateSpace`, used to generate candidate models.
    criterion:
        The criterion by which models will be compared.
    criterion_threshold:
        The minimum improvement in `criterion` that a test model must have to
        be selected. The comparison is made according to the method. For
        example, in `ForwardSelector`, test models are compared to the
        previously selected model.
    calibrated_models:
        All calibrated models of the model selection.
    limit:
        Limit the number of calibrated models. NB: the number of accepted
        models may (likely) be fewer.
    logger:
        A :class:`MethodLogger`, used to log results.
    predecessor_model:
        Specify the predecessor (initial) model for the model selection
        algorithm. If ``None``, then the algorithm will generate an initial
        predecessor model if required.
    user_calibrated_models:
        Supply calibration results for models yourself, as a list of models.
        If a model with the same hash is encountered in the current model
        selection run, and the user-supplied calibrated model has the
        `criterion` value set, the model will not be calibrated again.
    select_first_improvement:
        If ``True``, model selection will terminate as soon as a better model
        is found. If `False`, all candidate models will be tested.
    startpoint_latest_mle:
        If ``True``, one of the startpoints in the multistart optimization
        will be the MLE of the latest model.
    """

    def __init__(
        self,
        petab_select_problem: petab_select.Problem,
        calibrated_models: Models,
        # Arguments/attributes that can simply take the default value here.
        criterion_threshold: float = 0.0,
        limit: int = np.inf,
        # TODO deprecated
        minimize_options: dict = None,
        # TODO deprecated
        model_postprocessor: TYPE_POSTPROCESSOR = None,
        # TODO deprecated
        objective_customizer: Callable = None,
        select_first_improvement: bool = False,
        startpoint_latest_mle: bool = True,
        # Arguments/attributes that should be handled more carefully.
        candidate_space: CandidateSpace = None,
        criterion: Criterion = None,
        # TODO misleading, `Method` here is simply an Enum, not a callable...
        method: Method = None,
        predecessor_model: Model = None,
        # TODO deprecated
        model_to_pypesto_problem_method: Callable[[Any], Problem] = None,
        model_problem_options: dict = None,
        user_calibrated_models: list[Model] = None,
    ):
        """Arguments are used in every `__call__`, unless overridden."""
        self.petab_select_problem = petab_select_problem
        self.calibrated_models = calibrated_models

        self.criterion_threshold = criterion_threshold
        self.limit = limit
        self.predecessor_model = predecessor_model
        self.select_first_improvement = select_first_improvement
        self.startpoint_latest_mle = startpoint_latest_mle

        self.user_calibrated_models = Models()
        if user_calibrated_models is not None:
            self.user_calibrated_models = user_calibrated_models

        self.logger = MethodLogger()

        # TODO deprecated
        old_model_problem_options = {}
        for key, value in [
            ("postprocessor", model_postprocessor),
            (
                "model_to_pypesto_problem_method",
                model_to_pypesto_problem_method,
            ),
            ("minimize_options", minimize_options),
            ("objective_customizer", objective_customizer),
        ]:
            if value is not None:
                old_model_problem_options[key] = value
                self.logger.log(
                    f"Specifying `{key}` as an individual argument is "
                    "deprecated. Please instead specify it within some "
                    "`model_problem_options` dictionary, e.g. "
                    f'`model_problem_options={{"{key}": ...}}`.',
                    level="warning",
                )
        self.model_problem_options = {}
        self.model_problem_options |= old_model_problem_options
        if model_problem_options is not None:
            self.model_problem_options |= model_problem_options

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = self.petab_select_problem.criterion

        # Forbid specification of both a candidate space and a method.
        if candidate_space is not None and method is not None:
            self.logger.log(
                (
                    "Both `candidate_space` and `method` were provided. "
                    "Please only provide one. The method will be ignored here."
                ),
                level="warning",
            )
        # Get method.
        self.method = (
            method
            if method is not None
            else (
                candidate_space.method
                if candidate_space is not None
                else self.petab_select_problem.method
            )
        )
        # Require either a candidate space or a method.
        if candidate_space is None and self.method is None:
            raise ValueError(
                "Please provide one of either `candidate_space` or `method`, "
                "or specify the `method` in the PEtab Select problem."
            )
        # Use candidate space if provided.
        if candidate_space is not None:
            self.candidate_space = candidate_space
            if predecessor_model is not None:
                candidate_space.set_predecessor_model(predecessor_model)
        # Else generate one based on the PEtab Select problem.
        else:
            if predecessor_model is not None:
                self.candidate_space = (
                    self.petab_select_problem.new_candidate_space(
                        method=self.method,
                        predecessor_model=self.predecessor_model,
                    )
                )
            else:
                self.candidate_space = (
                    self.petab_select_problem.new_candidate_space(
                        method=self.method,
                    )
                )
        # May have changed from `None` to `petab_select.VIRTUAL_INITIAL_MODEL`
        self.predecessor_model = self.candidate_space.get_predecessor_model()

    def __call__(self) -> tuple[Model, Models]:
        """Run a single iteration of the model selection method.

        A single iteration here refers to calibration of all candidate models.
        For example, given a predecessor model with 3 estimated parameters,
        with the forward method, a single iteration would involve calibration
        of all models that have both: the same 3 estimated parameters; and 1
        additional estimated parameter.

        Returns
        -------
        A 2-tuple, with the following values:

           1. the predecessor model for the newly calibrated models; and
           2. the newly calibrated models.
        """
        # All calibrated models in this iteration (see second return value).
        self.logger.new_selection()

        iteration = petab_select.ui.start_iteration(
            problem=self.petab_select_problem,
            candidate_space=self.candidate_space,
            limit=self.limit,
            criterion=self.criterion,
            user_calibrated_models=self.user_calibrated_models,
        )

        if not iteration[UNCALIBRATED_MODELS]:
            raise StopIteration("No valid models found.")

        # TODO parallelize calibration (maybe not sensible if
        #      `self.select_first_improvement`)
        calibrated_models = Models()
        for model in iteration[UNCALIBRATED_MODELS]:
            if (
                model.get_criterion(
                    criterion=self.criterion,
                    compute=True,
                    raise_on_failure=False,
                )
                is not None
            ):
                self.logger.log(
                    message=(
                        "Unexpected calibration result already available for "
                        f"model: `{model.get_hash()}`. Skipping "
                        "calibration."
                    ),
                    level="warning",
                )
            else:
                self.new_model_problem(model=model)

            calibrated_models.append(model)
            method_signal = self.handle_calibrated_model(
                model=model,
                predecessor_model=iteration[PREDECESSOR_MODEL],
            )
            if method_signal.proceed == MethodSignalProceed.STOP:
                break

        iteration_results = petab_select.ui.end_iteration(
            problem=self.petab_select_problem,
            candidate_space=iteration[CANDIDATE_SPACE],
            calibrated_models=calibrated_models,
        )

        self.calibrated_models += iteration_results[MODELS]

        return iteration_results[MODELS]

    def handle_calibrated_model(
        self,
        model: Model,
        predecessor_model: Model,
    ) -> MethodSignal:
        """Handle the model selection method, given a new calibrated model.

        Parameters
        ----------
        model:
            The calibrated model.
        predecessor_model:
            The predecessor model.

        Returns
        -------
        A :class:`MethodSignal` that describes the result.
        """
        # Use the predecessor model from `__init__` if an iteration-specific
        # predecessor model was not supplied to `__call__`.
        if predecessor_model is None:
            # May still be `None` after this assignment.
            predecessor_model = self.predecessor_model

        # Default to accepting the model and continuing the method.
        method_signal = MethodSignal(
            accept=True,
            proceed=MethodSignalProceed.CONTINUE,
        )

        # Reject the model if it doesn't improve on the predecessor model.
        if (
            predecessor_model.hash != VIRTUAL_INITIAL_MODEL.hash
            and not self.model1_gt_model0(
                model1=model, model0=predecessor_model
            )
        ):
            method_signal.accept = False

        # Stop the model selection method if it a first improvement is found.
        if self.select_first_improvement and method_signal.accept:
            method_signal.proceed = MethodSignalProceed.STOP

        # TODO allow users to supply an arbitrary constraint function to e.g.:
        #      - quit after 10 accepted models
        #      - reject models that are worse than the current 10 best models

        # Log result
        self.logger.new_result(
            accept=method_signal.accept,
            criterion=self.criterion,
            model=model,
            predecessor_model=predecessor_model,
        )

        return method_signal

    def model1_gt_model0(
        self,
        model1: Model,
        model0: Model,
    ) -> bool:
        """Compare models by criterion.

        Parameters
        ----------
        model1:
            The new model.
        model0:
            The original model.

        Returns
        -------
        ``True``, if `model1` is superior to `model0` by the criterion,
        else ``False``.
        """
        if self.criterion in [
            Criterion.AIC,
            Criterion.AICC,
            Criterion.BIC,
            Criterion.LH,
            Criterion.LLH,
            Criterion.NLLH,
        ]:
            result = petab_select.model.default_compare(
                model0=model0,
                model1=model1,
                criterion=self.criterion,
                criterion_threshold=self.criterion_threshold,
            )
        else:
            raise NotImplementedError(
                f"Model selection criterion: {self.criterion}."
            )
        return result

    def new_model_problem(
        self,
        model: Model,
        valid: bool = True,
        autorun: bool = True,
    ) -> ModelProblem:
        """Create a model problem, usually to calibrate a model.

        Parameters
        ----------
        model:
            The model.
        valid:
            Whether the model should be considered a valid model. If it is
            not valid, it will not be calibrated.
        autorun:
            Whether the model should be calibrated upon creation.

        Returns
        -------
        The model selection problem.
        """
        x_guess = None
        if (
            self.startpoint_latest_mle
            and model.predecessor_model_hash in self.calibrated_models
        ):
            predecessor_model = self.calibrated_models[
                model.predecessor_model_hash
            ]
            if str(model.model_subspace_petab_yaml) != str(
                predecessor_model.model_subspace_petab_yaml
            ):
                raise NotImplementedError(
                    "The PEtab YAML files differ between the model and its "
                    "predecessor model. This may imply different (fixed union "
                    "estimated) parameter sets. Support for this is not yet "
                    "implemented."
                )
            x_guess = {
                **predecessor_model.parameters,
                **predecessor_model.estimated_parameters,
            }

        return ModelProblem(
            model=model,
            criterion=self.criterion,
            valid=valid,
            autorun=autorun,
            x_guess=x_guess,
            **self.model_problem_options,
        )
