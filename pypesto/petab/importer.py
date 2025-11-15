"""Contains the PetabImporter class."""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from collections.abc import Callable, Iterable, Sequence
from importlib.metadata import version
from typing import (
    Any,
)

import pandas as pd
import petab.v1 as petab

try:
    import roadrunner
except ImportError:
    roadrunner = None

from ..C import (
    AMICI,
    CENSORED,
    ORDINAL,
    ORDINAL_OPTIONS,
    PETAB,
    ROADRUNNER,
    SEMIQUANTITATIVE,
    SPLINE_APPROXIMATION_OPTIONS,
)
from ..hierarchical.inner_calculator_collector import InnerCalculatorCollector
from ..objective import AggregatedObjective, AmiciObjective, ObjectiveBase
from ..objective.priors import NegLogParameterPriors, get_parameter_prior_dict
from ..predict import AmiciPredictor
from ..problem import HierarchicalProblem, Problem
from ..result import PredictionResult
from ..startpoint import StartpointMethod
from .objective_creator import (
    AmiciObjectiveCreator,
    ObjectiveCreator,
    PetabSimulatorObjectiveCreator,
    RoadRunnerObjectiveCreator,
)
from .util import PetabStartpoints, get_petab_non_quantitative_data_types

try:
    import amici
    import amici.petab.simulations
except ImportError:
    amici = None

logger = logging.getLogger(__name__)


class PetabImporter:
    """
    Importer for PEtab files.

    Create an :class:`amici.amici.Model`, an :class:`pypesto.objective.AmiciObjective` or a
    :class:`pypesto.problem.Problem` from PEtab files. The created objective function is a
    negative log-likelihood function and can thus be negative. The actual
    form of the likelihood depends on the noise model specified in the provided PEtab problem.
    For more information, see the
    `PEtab documentation <https://petab.readthedocs.io/en/latest/documentation_data_format.html#noise-distributions>`_.
    """  # noqa

    MODEL_BASE_DIR = f"amici_models/{version('amici') if amici else ''}"

    def __init__(
        self,
        petab_problem: petab.Problem,
        output_folder: str | None = None,
        model_name: str | None = None,
        validate_petab: bool = True,
        validate_petab_hierarchical: bool = True,
        hierarchical: bool = False,
        inner_options: dict | None = None,
        simulator_type: str = AMICI,
        simulator: petab.Simulator | None = None,
        rr: roadrunner.RoadRunner | None = None,
    ):
        """Initialize importer.

        Parameters
        ----------
        petab_problem:
            Managing access to the model and data.
        output_folder:
            Folder to contain the amici model. Defaults to
            './amici_models/{model_name}'.
        model_name:
            Name of the model, which will in particular be the name of the
            compiled model python module.
        validate_petab:
            Flag indicating if the PEtab problem shall be validated.
        validate_petab_hierarchical:
            Flag indicating if the PEtab problem shall be validated in terms of
            pyPESTO's hierarchical optimization implementation.
        hierarchical:
            Whether to use hierarchical optimization or not, in case the
            underlying PEtab problem has parameters marked for hierarchical
            optimization (non-empty `parameterType` column in the PEtab
            parameter table). Required for ordinal, censored and semiquantitative data.
        inner_options:
            Options for the inner problems and solvers.
            If not provided, default options will be used.
        simulator_type:
            The type of simulator to use. Depending on this different kinds
            of objectives will be created. Allowed types are 'amici', 'petab',
            and 'roadrunner'.
        simulator:
            In case of a ``simulator_type == 'petab'``, the simulator object
            has to be provided. Otherwise, the argument is not used.
        """
        self.petab_problem = petab_problem
        self._hierarchical = hierarchical

        self._non_quantitative_data_types = (
            get_petab_non_quantitative_data_types(petab_problem)
        )

        if self._non_quantitative_data_types is None and hierarchical:
            raise ValueError(
                "Hierarchical optimization enabled, but no non-quantitative "
                "data types specified. Specify non-quantitative data types "
                "or disable hierarchical optimization."
            )

        if (
            self._non_quantitative_data_types is not None
            and any(
                data_type in self._non_quantitative_data_types
                for data_type in [ORDINAL, CENSORED, SEMIQUANTITATIVE]
            )
            and not self._hierarchical
        ):
            raise ValueError(
                "Ordinal, censored and semiquantitative data require "
                "hierarchical optimization to be enabled.",
            )

        self.inner_options = inner_options
        if self.inner_options is None:
            self.inner_options = {}

        self.validate_inner_options()

        self.validate_petab = validate_petab
        if self.validate_petab:
            if petab.lint_problem(petab_problem):
                raise ValueError("Invalid PEtab problem.")
        if self._hierarchical and validate_petab_hierarchical:
            from ..hierarchical.petab import (
                validate_hierarchical_petab_problem,
            )

            validate_hierarchical_petab_problem(petab_problem)

        if output_folder is None:
            output_folder = _find_output_folder_name(
                self.petab_problem,
                model_name=model_name,
            )
        self.output_folder = output_folder

        if model_name is None:
            model_name = _find_model_name(self.output_folder)
        self.model_name = model_name

        self.simulator_type = simulator_type
        self.simulator = simulator
        if simulator_type == PETAB and simulator is None:
            raise ValueError(
                "A petab simulator object must be provided if the simulator "
                "type is 'petab'."
            )
        self.roadrunner_instance = rr

    @staticmethod
    def from_yaml(
        yaml_config: dict | str,
        output_folder: str = None,
        model_name: str = None,
        simulator_type: str = AMICI,
    ) -> PetabImporter:
        """Simplified constructor using a petab yaml file."""
        petab_problem = petab.Problem.from_yaml(yaml_config)

        return PetabImporter(
            petab_problem=petab_problem,
            output_folder=output_folder,
            model_name=model_name,
            simulator_type=simulator_type,
        )

    def validate_inner_options(self):
        """Validate the inner options."""
        for key in self.inner_options:
            if key not in ORDINAL_OPTIONS + SPLINE_APPROXIMATION_OPTIONS:
                raise ValueError(f"Unknown inner option {key}.")

    def check_gradients(
        self,
        *args,
        rtol: float = 1e-2,
        atol: float = 1e-3,
        mode: str | list[str] = None,
        multi_eps=None,
        **kwargs,
    ) -> bool:
        """
        Check if gradients match finite differences (FDs).

        Parameters
        ----------
        rtol: relative error tolerance
        atol: absolute error tolerance
        mode: function values or residuals
        objAbsoluteTolerance: absolute tolerance in sensitivity calculation
        objRelativeTolerance: relative tolerance in sensitivity calculation
        multi_eps: multiple test step width for FDs

        Returns
        -------
        match: Whether gradients match FDs (True) or not (False)
        """
        raise NotImplementedError(
            "This function has been removed. "
            "Please use `objective.check_gradients_match_finite_differences`."
        )

    def create_prior(self) -> NegLogParameterPriors | None:
        """
        Create a prior from the parameter table.

        Returns None, if no priors are defined.
        """
        prior_list = []

        if petab.OBJECTIVE_PRIOR_TYPE in self.petab_problem.parameter_df:
            for i, x_id in enumerate(self.petab_problem.x_ids):
                prior_type_entry = self.petab_problem.parameter_df.loc[
                    x_id, petab.OBJECTIVE_PRIOR_TYPE
                ]

                if (
                    isinstance(prior_type_entry, str)
                    and prior_type_entry != petab.PARAMETER_SCALE_UNIFORM
                ):
                    # check if parameter for which prior is defined is a fixed parameter
                    if x_id in self.petab_problem.x_fixed_ids:
                        logger.warning(
                            f"Parameter {x_id} is marked as fixed but has a "
                            f"prior defined. This might be unintended."
                        )

                    prior_params = [
                        float(param)
                        for param in self.petab_problem.parameter_df.loc[
                            x_id, petab.OBJECTIVE_PRIOR_PARAMETERS
                        ].split(";")
                    ]

                    scale = self.petab_problem.parameter_df.loc[
                        x_id, petab.PARAMETER_SCALE
                    ]

                    prior_list.append(
                        get_parameter_prior_dict(
                            i, prior_type_entry, prior_params, scale
                        )
                    )

        if len(prior_list):
            return NegLogParameterPriors(prior_list)
        else:
            return None

    def create_startpoint_method(self, **kwargs) -> StartpointMethod:
        """Create a startpoint method.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments passed on to
            :meth:`pypesto.startpoint.FunctionStartpoints.__init__`.
        """
        return PetabStartpoints(petab_problem=self.petab_problem, **kwargs)

    def create_objective_creator(
        self,
        simulator_type: str = AMICI,
        simulator: petab.Simulator | None = None,
    ) -> ObjectiveCreator:
        """Choose :class:`ObjectiveCreator` depending on the simulator type.

        Parameters
        ----------
        simulator_type:
            The type of simulator to use. Depending on this different kinds
            of objectives will be created. Allowed types are 'amici', 'petab',
            and 'roadrunner'.
        simulator:
            In case of a ``simulator_type == 'petab'``, the simulator object
            has to be provided. Otherwise the argument is not used.

        """
        if simulator_type == AMICI:
            return AmiciObjectiveCreator(
                petab_problem=self.petab_problem,
                output_folder=self.output_folder,
                model_name=self.model_name,
                hierarchical=self._hierarchical,
                inner_options=self.inner_options,
                non_quantitative_data_types=self._non_quantitative_data_types,
                validate_petab=self.validate_petab,
            )
        elif simulator_type == PETAB:
            return PetabSimulatorObjectiveCreator(
                petab_problem=self.petab_problem, simulator=simulator
            )
        elif simulator_type == ROADRUNNER:
            return RoadRunnerObjectiveCreator(
                petab_problem=self.petab_problem, rr=self.roadrunner_instance
            )

    def create_problem(
        self,
        objective: ObjectiveBase = None,
        x_guesses: Iterable[float] | None = None,
        problem_kwargs: dict[str, Any] = None,
        startpoint_kwargs: dict[str, Any] = None,
        **kwargs,
    ) -> Problem:
        """Create a :class:`pypesto.problem.Problem`.

        Parameters
        ----------
        objective:
            Objective as created by :meth:`create_objective`.
        x_guesses:
            Guesses for the parameter values, shape (g, dim), where g denotes
            the number of guesses. These are used as start points in the
            optimization.
        problem_kwargs:
            Passed to :meth:`pypesto.problem.Problem.__init__`.
        startpoint_kwargs:
            Keyword arguments forwarded to
            :meth:`PetabImporter.create_startpoint_method`.
        **kwargs:
            Additional key word arguments passed on to the objective,
            if not provided.

        Returns
        -------
        A :class:`pypesto.problem.Problem` for the objective.
        """
        if objective is None:
            self.objective_constructor = self.create_objective_creator(
                kwargs.pop("simulator_type", self.simulator_type),
                kwargs.pop("simulator", self.simulator),
            )
            objective = self.objective_constructor.create_objective(**kwargs)

        x_fixed_indices = self.petab_problem.x_fixed_indices
        x_fixed_vals = self.petab_problem.x_nominal_fixed_scaled
        x_ids = self.petab_problem.x_ids
        lb = self.petab_problem.lb_scaled
        ub = self.petab_problem.ub_scaled

        # Raise error if the correct calculator is not used.
        if self._hierarchical:
            if not isinstance(objective.calculator, InnerCalculatorCollector):
                raise AssertionError(
                    f"If hierarchical optimization is enabled, the `calculator` attribute of the `objective` has to be {InnerCalculatorCollector} and not {objective.calculator}."
                )

        # In case of hierarchical optimization, parameters estimated in the
        # inner subproblem are removed from the outer problem
        if self._hierarchical:
            inner_parameter_ids = objective.calculator.get_inner_par_ids()
            lb = [b for x, b in zip(x_ids, lb) if x not in inner_parameter_ids]
            ub = [b for x, b in zip(x_ids, ub) if x not in inner_parameter_ids]
            x_ids = [x for x in x_ids if x not in inner_parameter_ids]
            x_fixed_indices = list(
                map(x_ids.index, self.petab_problem.x_fixed_ids)
            )

        x_scales = [
            self.petab_problem.parameter_df.loc[x_id, petab.PARAMETER_SCALE]
            for x_id in x_ids
        ]

        if problem_kwargs is None:
            problem_kwargs = {}

        if startpoint_kwargs is None:
            startpoint_kwargs = {}

        prior = self.create_prior()

        if prior is not None:
            if self._hierarchical:
                raise NotImplementedError(
                    "Hierarchical optimization in combination with priors "
                    "is not yet supported."
                )
            objective = AggregatedObjective([objective, prior])

        if self._hierarchical:
            problem_class = HierarchicalProblem
        else:
            problem_class = Problem

        problem = problem_class(
            objective=objective,
            lb=lb,
            ub=ub,
            x_fixed_indices=x_fixed_indices,
            x_fixed_vals=x_fixed_vals,
            x_guesses=x_guesses,
            x_names=x_ids,
            x_scales=x_scales,
            x_priors_defs=prior,
            startpoint_method=self.create_startpoint_method(
                **startpoint_kwargs
            ),
            **problem_kwargs,
        )

        return problem

    def create_model(
        self,
        force_compile: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> amici.Model:
        """See :meth:`AmiciObjectiveCreator.create_model`."""
        warnings.warn(
            "This function has been moved to `AmiciObjectiveCreator`.",
            DeprecationWarning,
            stacklevel=2,
        )
        objective_constructor = self.create_objective_creator(
            kwargs.pop("simulator_type", self.simulator_type),
            kwargs.pop("simulator", self.simulator),
        )
        return objective_constructor.create_model(
            force_compile=force_compile,
            verbose=verbose,
            **kwargs,
        )

    def create_objective(
        self,
        model: amici.Model = None,
        solver: amici.Solver = None,
        edatas: Sequence[amici.ExpData] = None,
        force_compile: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> ObjectiveBase:
        """See :meth:`AmiciObjectiveCreator.create_objective`."""
        warnings.warn(
            "This function has been moved to `AmiciObjectiveCreator`.",
            DeprecationWarning,
            stacklevel=2,
        )
        objective_constructor = self.create_objective_creator(
            kwargs.pop("simulator_type", self.simulator_type),
            kwargs.pop("simulator", self.simulator),
        )
        return objective_constructor.create_objective(
            model=model,
            solver=solver,
            edatas=edatas,
            force_compile=force_compile,
            verbose=verbose,
            **kwargs,
        )

    def create_predictor(
        self,
        objective: AmiciObjective = None,
        amici_output_fields: Sequence[str] = None,
        post_processor: Callable | None = None,
        post_processor_sensi: Callable | None = None,
        post_processor_time: Callable | None = None,
        max_chunk_size: int | None = None,
        output_ids: Sequence[str] = None,
        condition_ids: Sequence[str] = None,
    ) -> AmiciPredictor:
        """See :meth:`AmiciObjectiveCreator.create_predictor`."""
        if self.simulator_type != AMICI:
            raise ValueError(
                "Predictor can only be created for amici models and is "
                "supposed to be created from the AmiciObjectiveCreator."
            )
        warnings.warn(
            "This function has been moved to `AmiciObjectiveCreator`.",
            DeprecationWarning,
            stacklevel=2,
        )
        objective_constructor = self.create_objective_creator()
        return objective_constructor.create_predictor(
            objective=objective,
            amici_output_fields=amici_output_fields,
            post_processor=post_processor,
            post_processor_sensi=post_processor_sensi,
            post_processor_time=post_processor_time,
            max_chunk_size=max_chunk_size,
            output_ids=output_ids,
            condition_ids=condition_ids,
        )

    def rdatas_to_measurement_df(
        self,
        rdatas: Sequence[amici.ReturnData],
        model: amici.Model = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """See :meth:`AmiciObjectiveCreator.rdatas_to_measurement_df`."""
        raise NotImplementedError(
            "This function has been moved to `AmiciObjectiveCreator`."
        )

    def rdatas_to_simulation_df(
        self,
        rdatas: Sequence[amici.ReturnData],
        model: amici.Model = None,
    ) -> pd.DataFrame:
        """
        See :meth:`rdatas_to_measurement_df`.

        Except a petab simulation dataframe is created, i.e. the measurement
        column label is adjusted.
        """
        raise NotImplementedError(
            "This function has been moved to `AmiciObjectiveCreator`."
        )

    def prediction_to_petab_measurement_df(
        self,
        prediction: PredictionResult,
        predictor: AmiciPredictor = None,
    ) -> pd.DataFrame:
        """
        Cast prediction into a dataframe.

        If a PEtab problem is simulated without post-processing, then the
        result can be cast into a PEtab measurement or simulation dataframe

        Parameters
        ----------
        prediction:
            A prediction result as produced by an :class:`pypesto.predict.AmiciPredictor`.
        predictor:
            The :class:`pypesto.predict.AmiciPredictor` instance.

        Returns
        -------
        A dataframe built from the rdatas in the format as in
        ``self.petab_problem.measurement_df``.
        """
        raise NotImplementedError(
            "This function has been moved to `AmiciObjectiveCreator`."
        )

    def prediction_to_petab_simulation_df(
        self,
        prediction: PredictionResult,
        predictor: AmiciPredictor = None,
    ) -> pd.DataFrame:
        """
        See :meth:`prediction_to_petab_measurement_df`.

        Except a PEtab simulation dataframe is created, i.e. the measurement
        column label is adjusted.
        """
        raise NotImplementedError(
            "This function has been moved to `AmiciObjectiveCreator`."
        )


def _find_output_folder_name(
    petab_problem: petab.Problem,
    model_name: str,
) -> str:
    """
    Find a name for storing the compiled amici model in.

    If available, use the model name from the ``petab_problem`` or the
    provided ``model_name`` (latter is given priority), otherwise create a
    unique name. The folder will be located in the
    :obj:`PetabImporter.MODEL_BASE_DIR` subdirectory of the current directory.
    """
    # check whether location for amici model is a file
    if os.path.exists(PetabImporter.MODEL_BASE_DIR) and not os.path.isdir(
        PetabImporter.MODEL_BASE_DIR
    ):
        raise AssertionError(
            f"{PetabImporter.MODEL_BASE_DIR} exists and is not a directory, "
            f"thus cannot create a directory for the compiled amici model."
        )

    # create base directory if non-existent
    if not os.path.exists(PetabImporter.MODEL_BASE_DIR):
        os.makedirs(PetabImporter.MODEL_BASE_DIR)

    # try model id
    model_id = petab_problem.model.model_id
    if model_name is not None:
        model_id = model_name

    if model_id:
        output_folder = os.path.abspath(
            os.path.join(PetabImporter.MODEL_BASE_DIR, model_id)
        )
    else:
        # create random folder name
        output_folder = os.path.abspath(
            tempfile.mkdtemp(dir=PetabImporter.MODEL_BASE_DIR)
        )
    return output_folder


def _find_model_name(output_folder: str) -> str:
    """Just re-use the last part of the output folder."""
    return os.path.split(os.path.normpath(output_folder))[-1]
