"""Contains the PetabImporter class."""
from __future__ import annotations

import importlib
import logging
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pandas as pd

from ..C import (
    CENSORED,
    CENSORING_TYPES,
    CONDITION_SEP,
    MEASUREMENT_TYPE,
    MODE_FUN,
    MODE_RES,
    NONLINEAR_MONOTONE,
    OPTIMAL_SCALING_OPTIONS,
    ORDINAL,
    SPLINE_APPROXIMATION_OPTIONS,
)
from ..hierarchical.calculator import HierarchicalAmiciCalculator
from ..hierarchical.inner_calculator_collector import InnerCalculatorCollector
from ..hierarchical.problem import InnerProblem
from ..objective import AggregatedObjective, AmiciObjective
from ..objective.amici import AmiciObjectBuilder
from ..objective.priors import NegLogParameterPriors, get_parameter_prior_dict
from ..predict import AmiciPredictor
from ..problem import Problem
from ..result import PredictionResult
from ..startpoint import FunctionStartpoints, StartpointMethod

try:
    import amici
    import amici.parameter_mapping
    import amici.petab_import
    import amici.petab_objective
    import petab
    from petab.C import PREEQUILIBRATION_CONDITION_ID, SIMULATION_CONDITION_ID
except ImportError:
    pass

logger = logging.getLogger(__name__)


class PetabImporter(AmiciObjectBuilder):
    """
    Importer for PEtab files.

    Create an `amici.Model`, an `objective.AmiciObjective` or a
    `pypesto.Problem` from PEtab files. The created objective function is a
    negative log-likelihood function and can thus be negative. The actual
    form of the likelihood depends on the noise model specified in the provided PEtab problem.
    For more information, see
    [the PEtab documentation](https://petab.readthedocs.io/en/latest/documentation_data_format.html#noise-distributions)
    """  # noqa

    MODEL_BASE_DIR = "amici_models"

    def __init__(
        self,
        petab_problem: petab.Problem,
        output_folder: str = None,
        model_name: str = None,
        validate_petab: bool = True,
        validate_petab_hierarchical: bool = True,
        hierarchical: bool = False,
        inner_options: Dict = None,
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
            parameter table). Required for ordinal, censored and nonlinear-monotone data.
        inner_options:
            Options for the inner problems and solvers.
            If not provided, default options will be used.
        """
        self.petab_problem = petab_problem
        self._hierarchical = hierarchical

        self._non_quantitative_data_types = (
            get_petab_non_quantitative_data_types(petab_problem)
        )

        if self._non_quantitative_data_types and not self._hierarchical:
            raise ValueError(
                "Ordinal, censored and nonlinear-monotone data require "
                "hierarchical optimization to be enabled.",
            )

        self.inner_options = inner_options
        if self.inner_options is None:
            self.inner_options = {}

        self.validate_inner_options()

        if validate_petab:
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
        yaml_config: Union[dict, str],
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
            if (
                key
                not in OPTIMAL_SCALING_OPTIONS + SPLINE_APPROXIMATION_OPTIONS
            ):
                raise ValueError(f"Unknown inner option {key}.")

    def check_gradients(
        self,
        *args,
        rtol: float = 1e-2,
        atol: float = 1e-3,
        mode: Union[str, List[str]] = None,
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
        modes = []

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

    def create_model(
        self,
        force_compile: bool = False,
        **kwargs,
    ) -> amici.Model:
        """
        Import amici model.

        Parameters
        ----------
        force_compile:
            If False, the model is compiled only if the output folder does not
            exist yet. If True, the output folder is deleted and the model
            (re-)compiled in either case.

            .. warning::
                If `force_compile`, then an existing folder of that name will
                be deleted.

        kwargs: Extra arguments passed to amici.SbmlImporter.sbml2amici
        """
        # courtesy check whether target is folder
        if os.path.exists(self.output_folder) and not os.path.isdir(
            self.output_folder
        ):
            raise AssertionError(
                f"Refusing to remove {self.output_folder} for model "
                f"compilation: Not a folder."
            )

        # add module to path
        if self.output_folder not in sys.path:
            sys.path.insert(0, self.output_folder)

        # compile
        if self._must_compile(force_compile):
            logger.info(
                f"Compiling amici model to folder " f"{self.output_folder}."
            )
            self.compile_model(**kwargs)
        else:
            logger.debug(
                f"Using existing amici model in folder "
                f"{self.output_folder}."
            )

        return self._create_model()

    def _create_model(self) -> amici.Model:
        """Load model module and return the model, no checks/compilation."""
        # load moduĺe
        module = amici.import_model_module(
            module_name=self.model_name, module_path=self.output_folder
        )
        model = module.getModel()
        amici.petab_import.check_model(
            amici_model=model,
            petab_problem=self.petab_problem,
        )

        return model

    def _must_compile(self, force_compile: bool):
        """Check whether the model needs to be compiled first."""
        # asked by user
        if force_compile:
            return True

        # folder does not exist
        if not os.path.exists(self.output_folder) or not os.listdir(
            self.output_folder
        ):
            return True

        # try to import (in particular checks version)
        try:
            # importing will already raise an exception if version wrong
            importlib.import_module(self.model_name)
        except ModuleNotFoundError:
            return True
        except amici.AmiciVersionError as e:
            logger.info(
                "amici model will be re-imported due to version "
                f"mismatch: {e}"
            )
            return True

        # no need to (re-)compile
        return False

    def compile_model(self, **kwargs):
        """
        Compile the model.

        If the output folder exists already, it is first deleted.

        Parameters
        ----------
        kwargs:
            Extra arguments passed to :meth:`amici.SbmlImporter.sbml2amici`
            or :meth:`amici.pysb_import.pysb2amici`.
        """
        # delete output directory
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        amici.petab_import.import_petab_problem(
            petab_problem=self.petab_problem,
            model_name=self.model_name,
            model_output_dir=self.output_folder,
            **kwargs,
        )

    def create_solver(self, model: amici.Model = None) -> amici.Solver:
        """Return model solver."""
        # create model
        if model is None:
            model = self.create_model()

        solver = model.getSolver()
        return solver

    def create_edatas(
        self, model: amici.Model = None, simulation_conditions=None
    ) -> List[amici.ExpData]:
        """Create list of amici.ExpData objects."""
        # create model
        if model is None:
            model = self.create_model()

        return amici.petab_objective.create_edatas(
            amici_model=model,
            petab_problem=self.petab_problem,
            simulation_conditions=simulation_conditions,
        )

    def create_objective(
        self,
        model: amici.Model = None,
        solver: amici.Solver = None,
        edatas: Sequence[amici.ExpData] = None,
        force_compile: bool = False,
        **kwargs,
    ) -> AmiciObjective:
        """Create a :class:`pypesto.AmiciObjective`.

        Parameters
        ----------
        model:
            The AMICI model.
        solver:
            The AMICI solver.
        edatas:
            The experimental data in AMICI format.
        force_compile:
            Whether to force-compile the model if not passed.
        **kwargs:
            Additional arguments passed on to the objective. In case of ordinal
            or nonlinear-monotone measurements, ``inner_options`` can optionally
            be passed here. If none are given, ``inner_options`` given to the
            importer constructor (or inner defaults) will be chosen.

        Returns
        -------
        objective:
            A :class:`pypesto.AmiciObjective` for the model and the data.
        """
        # get simulation conditions
        simulation_conditions = petab.get_simulation_conditions(
            self.petab_problem.measurement_df
        )

        # create model
        if model is None:
            model = self.create_model(force_compile=force_compile)
        # create solver
        if solver is None:
            solver = self.create_solver(model)
        # create conditions and edatas from measurement data
        if edatas is None:
            edatas = self.create_edatas(
                model=model, simulation_conditions=simulation_conditions
            )

        parameter_mapping = amici.petab_objective.create_parameter_mapping(
            petab_problem=self.petab_problem,
            simulation_conditions=simulation_conditions,
            scaled_parameters=True,
            amici_model=model,
            fill_fixed_parameters=False,
        )

        par_ids = self.petab_problem.x_ids

        # fill in dummy parameters (this is needed since some objective
        #  initialization e.g. checks for preeq parameters)
        problem_parameters = dict(
            zip(self.petab_problem.x_ids, self.petab_problem.x_nominal_scaled)
        )
        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=problem_parameters,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=model,
        )

        calculator = None
        amici_reporting = None

        if self._non_quantitative_data_types and self._hierarchical:
            inner_options = kwargs.pop('inner_options', None)
            inner_options = (
                inner_options
                if inner_options is not None
                else self.inner_options
            )
            calculator = InnerCalculatorCollector(
                self._non_quantitative_data_types,
                self.petab_problem,
                model,
                edatas,
                inner_options,
            )
            amici_reporting = amici.RDataReporting.residuals

        elif self._hierarchical:
            inner_problem = InnerProblem.from_petab_amici(
                self.petab_problem, model, edatas
            )
            calculator = HierarchicalAmiciCalculator(inner_problem)
            amici_reporting = amici.RDataReporting.full

        if self._hierarchical:
            # FIXME: currently not supported with hierarchical
            if 'guess_steadystate' in kwargs and kwargs['guess_steadystate']:
                warnings.warn(
                    "`guess_steadystate` not supported with hierarchical optimization. Disabling `guess_steadystate`."
                )
            kwargs['guess_steadystate'] = False
            inner_parameter_ids = calculator.get_inner_parameter_ids()
            par_ids = [x for x in par_ids if x not in inner_parameter_ids]

        # create objective
        obj = AmiciObjective(
            amici_model=model,
            amici_solver=solver,
            edatas=edatas,
            x_ids=par_ids,
            x_names=par_ids,
            parameter_mapping=parameter_mapping,
            amici_object_builder=self,
            calculator=calculator,
            amici_reporting=amici_reporting,
            **kwargs,
        )

        return obj

    def create_predictor(
        self,
        objective: AmiciObjective = None,
        amici_output_fields: Sequence[str] = None,
        post_processor: Union[Callable, None] = None,
        post_processor_sensi: Union[Callable, None] = None,
        post_processor_time: Union[Callable, None] = None,
        max_chunk_size: Union[int, None] = None,
        output_ids: Sequence[str] = None,
        condition_ids: Sequence[str] = None,
    ) -> AmiciPredictor:
        """Create a :class:`pypesto.predict.AmiciPredictor`.

        The `AmiciPredictor` facilitates generation of predictions from
        parameter vectors.

        Parameters
        ----------
        objective:
            An objective object, which will be used to get model simulations
        amici_output_fields:
            keys that exist in the return data object from AMICI, which should
            be available for the post-processors
        post_processor:
            A callable function which applies postprocessing to the simulation
            results. Default are the observables of the AMICI model.
            This method takes a list of ndarrays (as returned in the field
            ['y'] of amici ReturnData objects) as input.
        post_processor_sensi:
            A callable function which applies postprocessing to the
            sensitivities of the simulation results. Default are the
            observable sensitivities of the AMICI model.
            This method takes two lists of ndarrays (as returned in the
            fields ['y'] and ['sy'] of amici ReturnData objects) as input.
        post_processor_time:
            A callable function which applies postprocessing to the timepoints
            of the simulations. Default are the timepoints of the amici model.
            This method takes a list of ndarrays (as returned in the field
            ['t'] of amici ReturnData objects) as input.
        max_chunk_size:
            In some cases, we don't want to compute all predictions at once
            when calling the prediction function, as this might not fit into
            the memory for large datasets and models.
            Here, the user can specify a maximum number of conditions, which
            should be simulated at a time.
            Default is 0 meaning that all conditions will be simulated.
            Other values are only applicable, if an output file is specified.
        output_ids:
            IDs of outputs, if post-processing is used
        condition_ids:
            IDs of conditions, if post-processing is used

        Returns
        -------
        predictor:
            A :class:`pypesto.predict.AmiciPredictor` for the model, using
            the outputs of the AMICI model and the timepoints from the
            PEtab data
        """
        # if the user didn't pass an objective function, we create it first
        if objective is None:
            objective = self.create_objective()

        # create a identifiers of preequilibration and simulation condition ids
        # which can then be stored in the prediction result
        edata_conditions = (
            objective.amici_object_builder.petab_problem.get_simulation_conditions_from_measurement_df()
        )
        if PREEQUILIBRATION_CONDITION_ID not in list(edata_conditions.columns):
            preeq_dummy = [''] * edata_conditions.shape[0]
            edata_conditions[PREEQUILIBRATION_CONDITION_ID] = preeq_dummy
        edata_conditions.drop_duplicates(inplace=True)

        if condition_ids is None:
            condition_ids = [
                edata_conditions.loc[id, PREEQUILIBRATION_CONDITION_ID]
                + CONDITION_SEP
                + edata_conditions.loc[id, SIMULATION_CONDITION_ID]
                for id in edata_conditions.index
            ]

        # wrap around AmiciPredictor
        predictor = AmiciPredictor(
            amici_objective=objective,
            amici_output_fields=amici_output_fields,
            post_processor=post_processor,
            post_processor_sensi=post_processor_sensi,
            post_processor_time=post_processor_time,
            max_chunk_size=max_chunk_size,
            output_ids=output_ids,
            condition_ids=condition_ids,
        )

        return predictor

    def create_prior(self) -> Union[NegLogParameterPriors, None]:
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
                    prior_params = [
                        float(param)
                        for param in self.petab_problem.parameter_df.loc[
                            x_id, petab.OBJECTIVE_PRIOR_PARAMETERS
                        ].split(';')
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

    def create_startpoint_method(
        self, x_ids: Sequence[str] = None, **kwargs
    ) -> StartpointMethod:
        """Create a startpoint method.

        Parameters
        ----------
        x_ids:
            If provided, create a startpoint method that only samples the
            parameters with the given IDs.
        **kwargs:
            Additional keyword arguments passed on to
            :meth:`pypesto.startpoint.FunctionStartpoints.__init__`.
        """

        def startpoint_method(n_starts: int, **kwargs):
            startpoints = petab.sample_parameter_startpoints(
                self.petab_problem.parameter_df, n_starts=n_starts
            )
            if x_ids is None:
                return startpoints

            # subset parameters according to the provided parameter IDs
            from petab.C import ESTIMATE

            parameter_df = self.petab_problem.parameter_df
            pars_to_estimate = list(
                parameter_df.index[parameter_df[ESTIMATE] == 1]
            )
            x_idxs = [pars_to_estimate.index(x_id) for x_id in x_ids]
            return startpoints[:, x_idxs]

        return FunctionStartpoints(function=startpoint_method, **kwargs)

    def create_problem(
        self,
        objective: AmiciObjective = None,
        x_guesses: Optional[Iterable[float]] = None,
        problem_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> Problem:
        """Create a :class:`pypesto.Problem`.

        Parameters
        ----------
        objective:
            Objective as created by `create_objective`.
        x_guesses:
            Guesses for the parameter values, shape (g, dim), where g denotes
            the number of guesses. These are used as start points in the
            optimization.
        problem_kwargs:
            Passed to the `pypesto.Problem` constructor.
        **kwargs:
            Additional key word arguments passed on to the objective,
            if not provided.

        Returns
        -------
        problem:
            A :class:`pypesto.Problem` for the objective.
        """
        if objective is None:
            objective = self.create_objective(**kwargs)

        x_fixed_indices = self.petab_problem.x_fixed_indices
        x_fixed_vals = self.petab_problem.x_nominal_fixed_scaled
        x_ids = self.petab_problem.x_ids
        lb = self.petab_problem.lb_scaled
        ub = self.petab_problem.ub_scaled

        # Raise error if the correct calculator is not used.

        if self._non_quantitative_data_types:
            if not isinstance(objective.calculator, InnerCalculatorCollector):
                raise AssertionError(
                    f"If there are ordinal, censored or nonlinear-monotone measurements, the `calculator` attribute of the `objective` has to be {InnerCalculatorCollector} and not {objective.calculator}."
                )
        elif self._hierarchical:
            if not isinstance(
                objective.calculator, HierarchicalAmiciCalculator
            ):
                raise AssertionError(
                    f"If hierarchical optimization of relative data is enabled, the `calculator` attribute of the `objective` has to be {HierarchicalAmiciCalculator} and not {objective.calculator}."
                )
        # In case of hierarchical optimization, parameters estimated in the
        # inner subproblem are removed from the outer problem
        if self._hierarchical:
            inner_parameter_ids = (
                objective.calculator.get_inner_parameter_ids()
            )
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

        prior = self.create_prior()

        if prior is not None:
            if self._hierarchical:
                raise NotImplementedError(
                    "Hierarchical optimization in combination with priors "
                    "is not yet supported."
                )
            objective = AggregatedObjective([objective, prior])

        problem = Problem(
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
                x_ids=np.delete(x_ids, x_fixed_indices)
            ),
            **problem_kwargs,
        )

        return problem

    def rdatas_to_measurement_df(
        self,
        rdatas: Sequence[amici.ReturnData],
        model: amici.Model = None,
    ) -> pd.DataFrame:
        """
        Create a measurement dataframe in the petab format.

        Parameters
        ----------
        rdatas:
            A list of rdatas as produced by
            pypesto.AmiciObjective.__call__(x, return_dict=True)['rdatas'].
        model:
            The amici model.

        Returns
        -------
        measurement_df:
            A dataframe built from the rdatas in the format as in
            self.petab_problem.measurement_df.
        """
        # create model
        if model is None:
            model = self.create_model()

        measurement_df = self.petab_problem.measurement_df

        return amici.petab_objective.rdatas_to_measurement_df(
            rdatas, model, measurement_df
        )

    def rdatas_to_simulation_df(
        self,
        rdatas: Sequence[amici.ReturnData],
        model: amici.Model = None,
    ) -> pd.DataFrame:
        """
        See `rdatas_to_measurement_df`.

        Execpt a petab simulation dataframe is created, i.e. the measurement
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
            A prediction result as produced by an AmiciPredictor
        predictor:
            The AmiciPredictor function

        Returns
        -------
        measurement_df:
            A dataframe built from the rdatas in the format as in
            self.petab_problem.measurement_df.
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
        See `prediction_to_petab_measurement_df`.

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
    ``PetabImporter.MODEL_BASE_DIR`` subdirectory of the current directory.
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
) -> List[str]:
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
    non_quantitative_data_types = []
    if MEASUREMENT_TYPE in petab_problem.measurement_df.columns:
        petab_data_types = petab_problem.measurement_df[
            MEASUREMENT_TYPE
        ].unique()
        if ORDINAL in petab_data_types:
            non_quantitative_data_types.append(ORDINAL)
        if any(
            censoring_type in petab_data_types
            for censoring_type in CENSORING_TYPES
        ):
            non_quantitative_data_types.append(CENSORED)
        if NONLINEAR_MONOTONE in petab_data_types:
            non_quantitative_data_types.append(NONLINEAR_MONOTONE)

    if len(non_quantitative_data_types) == 0:
        return None
    return non_quantitative_data_types
