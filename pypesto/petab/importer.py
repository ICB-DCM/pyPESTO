"""Contains the PetabImporter class."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version
from typing import (
    Any,
)

import numpy as np
import pandas as pd
import petab.v1 as petab
from petab.v1.C import (
    ESTIMATE,
    NOISE_PARAMETERS,
    OBSERVABLE_ID,
)

from ..C import (
    CENSORED,
    CENSORING_TYPES,
    MEASUREMENT_TYPE,
    MODE_FUN,
    MODE_RES,
    ORDINAL,
    ORDINAL_OPTIONS,
    PARAMETER_TYPE,
    RELATIVE,
    SEMIQUANTITATIVE,
    SPLINE_APPROXIMATION_OPTIONS,
    InnerParameterType,
)
from ..hierarchical.inner_calculator_collector import InnerCalculatorCollector
from ..objective import AggregatedObjective, AmiciObjective
from ..objective.priors import NegLogParameterPriors, get_parameter_prior_dict
from ..predict import AmiciPredictor
from ..problem import HierarchicalProblem, Problem
from ..result import PredictionResult
from ..startpoint import CheckedStartpoints, StartpointMethod
from .factory import AmiciFactory

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
        output_folder: str = None,
        model_name: str = None,
        validate_petab: bool = True,
        validate_petab_hierarchical: bool = True,
        hierarchical: bool = False,
        inner_options: dict = None,
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

    @staticmethod
    def from_yaml(
        yaml_config: dict | str,
        output_folder: str = None,
        model_name: str = None,
    ) -> PetabImporter:
        """Simplified constructor using a petab yaml file."""
        petab_problem = petab.Problem.from_yaml(yaml_config)

        return PetabImporter(
            petab_problem=petab_problem,
            output_folder=output_folder,
            model_name=model_name,
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
        par = np.asarray(self.petab_problem.x_nominal_scaled)
        problem = self.create_problem()
        objective = problem.objective
        free_indices = par[problem.x_free_indices]
        dfs = []

        if mode is None:
            modes = [MODE_FUN, MODE_RES]
        else:
            modes = [mode]

        if multi_eps is None:
            multi_eps = np.array([10 ** (-i) for i in range(3, 9)])

        for mode in modes:
            try:
                dfs.append(
                    objective.check_grad_multi_eps(
                        free_indices,
                        *args,
                        **kwargs,
                        mode=mode,
                        multi_eps=multi_eps,
                    )
                )
            except (RuntimeError, ValueError):
                # Might happen in case PEtab problem not well defined or
                # fails for specified tolerances in forward sensitivities
                return False

        return all(
            any(
                [
                    np.all(
                        (mode_df.rel_err.values < rtol)
                        | (mode_df.abs_err.values < atol)
                    ),
                ]
            )
            for mode_df in dfs
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

    def create_problem(
        self,
        objective: AmiciObjective = None,
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
            factory = AmiciFactory(
                petab_problem=self.petab_problem,
                output_folder=self.output_folder,
                model_name=self.model_name,
                hierarchical=self._hierarchical,
                inner_options=self.inner_options,
                non_quantitative_data_types=self._non_quantitative_data_types,
                validate_petab=self.validate_petab,
            )
            objective = factory.create_objective(**kwargs)

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

    def rdatas_to_measurement_df(
        self,
        rdatas: Sequence[amici.ReturnData],
        model: amici.Model = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Create a measurement dataframe in the petab format.

        Parameters
        ----------
        rdatas:
            A list of rdatas as produced by
            ``pypesto.AmiciObjective.__call__(x, return_dict=True)['rdatas']``.
        model:
            The amici model.
        verbose:
            Passed to AMICI's model compilation. If True, the compilation
            progress is printed.

        Returns
        -------
        A dataframe built from the rdatas in the format as in
        ``self.petab_problem.measurement_df``.
        """
        # create model
        if model is None:
            model = self.create_model(verbose=verbose)

        measurement_df = self.petab_problem.measurement_df

        return amici.petab.simulations.rdatas_to_measurement_df(
            rdatas, model, measurement_df
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
        return self.rdatas_to_measurement_df(rdatas, model).rename(
            columns={petab.MEASUREMENT: petab.SIMULATION}
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

        # create rdata-like dicts from the prediction result
        @dataclass
        class FakeRData:
            ts: np.ndarray
            y: np.ndarray

        rdatas = [
            FakeRData(ts=condition.timepoints, y=condition.output)
            for condition in prediction.conditions
        ]

        # add an AMICI model, if possible
        model = None
        if predictor is not None:
            model = predictor.amici_objective.amici_model

        return self.rdatas_to_measurement_df(rdatas, model)

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
        return self.prediction_to_petab_measurement_df(
            prediction, predictor
        ).rename(columns={petab.MEASUREMENT: petab.SIMULATION})


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


def get_petab_non_quantitative_data_types(
    petab_problem: petab.Problem,
) -> set[str]:
    """
    Get the data types from the PEtab problem.

    Parameters
    ----------
    petab_problem:
        The PEtab problem.

    Returns
    -------
    data_types:
        A list of the data types.
    """
    non_quantitative_data_types = set()
    caught_observables = set()
    # For ordinal, censored and semiquantitative data, search
    # for the corresponding data types in the measurement table
    meas_df = petab_problem.measurement_df
    if MEASUREMENT_TYPE in meas_df.columns:
        petab_data_types = meas_df[MEASUREMENT_TYPE].unique()
        for data_type in [ORDINAL, SEMIQUANTITATIVE] + CENSORING_TYPES:
            if data_type in petab_data_types:
                non_quantitative_data_types.add(
                    CENSORED if data_type in CENSORING_TYPES else data_type
                )
                caught_observables.update(
                    set(
                        meas_df[meas_df[MEASUREMENT_TYPE] == data_type][
                            OBSERVABLE_ID
                        ]
                    )
                )

    # For relative data, search for parameters to estimate with
    # a scaling/offset/sigma parameter type
    if PARAMETER_TYPE in petab_problem.parameter_df.columns:
        # get the df with non-nan parameter types
        par_df = petab_problem.parameter_df[
            petab_problem.parameter_df[PARAMETER_TYPE].notna()
        ]
        for par_id, row in par_df.iterrows():
            if not row[ESTIMATE]:
                continue
            if row[PARAMETER_TYPE] in [
                InnerParameterType.SCALING,
                InnerParameterType.OFFSET,
            ]:
                non_quantitative_data_types.add(RELATIVE)

            # For sigma parameters, we need to check if they belong
            # to an observable with a non-quantitative data type
            elif row[PARAMETER_TYPE] == InnerParameterType.SIGMA:
                corresponding_observables = set(
                    meas_df[meas_df[NOISE_PARAMETERS] == par_id][OBSERVABLE_ID]
                )
                if not (corresponding_observables & caught_observables):
                    non_quantitative_data_types.add(RELATIVE)

    # TODO this can be made much shorter if the relative measurements
    # are also specified in the measurement table, but that would require
    # changing the PEtab format of a lot of benchmark models.

    if len(non_quantitative_data_types) == 0:
        return None
    return non_quantitative_data_types


class PetabStartpoints(CheckedStartpoints):
    """Startpoint method for PEtab problems.

    Samples optimization startpoints from the distributions defined in the
    provided PEtab problem. The PEtab-problem is copied.
    """

    def __init__(self, petab_problem: petab.Problem, **kwargs):
        super().__init__(**kwargs)
        self._parameter_df = petab_problem.parameter_df.copy()
        self._priors: list[tuple] | None = None
        self._free_ids: list[str] | None = None

    def _setup(
        self,
        pypesto_problem: Problem,
    ):
        """Update priors if necessary.

        Check if ``problem.x_free_indices`` changed since last call, and if so,
        get the corresponding priors from PEtab.
        """
        current_free_ids = np.asarray(pypesto_problem.x_names)[
            pypesto_problem.x_free_indices
        ]

        if (
            self._priors is not None
            and len(current_free_ids) == len(self._free_ids)
            and np.all(current_free_ids == self._free_ids)
        ):
            # no need to update
            return

        # update priors
        self._free_ids = current_free_ids
        id_to_prior = dict(
            zip(
                self._parameter_df.index[self._parameter_df[ESTIMATE] == 1],
                petab.parameters.get_priors_from_df(
                    self._parameter_df, mode=petab.INITIALIZATION
                ),
            )
        )

        self._priors = list(map(id_to_prior.__getitem__, current_free_ids))

    def __call__(
        self,
        n_starts: int,
        problem: Problem,
    ) -> np.ndarray:
        """Call the startpoint method."""
        # Update the list of priors if needed
        self._setup(pypesto_problem=problem)

        return super().__call__(n_starts, problem)

    def sample(
        self,
        n_starts: int,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        """Actual startpoint sampling.

        Must only be called through `self.__call__` to ensure that the list of priors
        matches the currently free parameters in the :class:`pypesto.Problem`.
        """
        sampler = partial(petab.sample_from_prior, n_starts=n_starts)
        startpoints = list(map(sampler, self._priors))

        return np.array(startpoints).T
