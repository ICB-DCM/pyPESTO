"""Contains the ObjectiveCreator class."""

from __future__ import annotations

import logging
import numbers
import os
import re
import shutil
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
)

import numpy as np
import pandas as pd
import petab.v1 as petab
from petab.v1.C import (
    OBSERVABLE_FORMULA,
    PREEQUILIBRATION_CONDITION_ID,
    SIMULATION_CONDITION_ID,
)
from petab.v1.models import MODEL_TYPE_SBML
from petab.v1.models.sbml_model import SbmlModel
from petab.v1.parameter_mapping import ParMappingDictQuadruple
from petab.v1.simulate import Simulator

from ..C import CENSORED, CONDITION_SEP, LIN, ORDINAL, SEMIQUANTITATIVE
from ..hierarchical.inner_calculator_collector import InnerCalculatorCollector
from ..objective import AmiciObjective, ObjectiveBase, PetabSimulatorObjective
from ..objective.amici import AmiciObjectBuilder
from ..objective.roadrunner import (
    ExpData,
    RoadRunnerCalculator,
    RoadRunnerObjective,
)
from ..predict import AmiciPredictor
from ..result import PredictionResult

try:
    import amici
    import amici.petab
    import amici.petab.conditions
    import amici.petab.parameter_mapping
    import amici.petab.simulations
    from amici.petab.import_helpers import check_model
except ImportError:
    amici = None
try:
    import libsbml
    import roadrunner
except ImportError:
    roadrunner = None
    libsbml = None

logger = logging.getLogger(__name__)


class ObjectiveCreator(ABC):
    """Abstract Creator for creating an objective function."""

    @abstractmethod
    def create_objective(self, **kwargs) -> ObjectiveBase:
        """Create an objective function."""
        pass


class AmiciObjectiveCreator(ObjectiveCreator, AmiciObjectBuilder):
    """ObjectiveCreator for creating an amici objective function."""

    def __init__(
        self,
        petab_problem: petab.Problem,
        hierarchical: bool = False,
        non_quantitative_data_types: Iterable[str] | None = None,
        inner_options: dict[str, Any] | None = None,
        output_folder: str | None = None,
        model_name: str | None = None,
        validate_petab: bool = True,
    ):
        """
        Initialize the creator.

        Parameters
        ----------
        petab_problem:
            The PEtab problem.
        hierarchical:
            Whether to use hierarchical optimization.
        non_quantitative_data_types:
            The non-quantitative data types to consider.
        inner_options:
            Options for the inner optimization.
        output_folder:
            The output folder for the compiled model.
        model_name:
            The name of the model.
        validate_petab:
            Whether to check the PEtab problem for errors.
        """
        self.petab_problem = petab_problem
        self._hierarchical = hierarchical
        self._non_quantitative_data_types = non_quantitative_data_types
        self.inner_options = inner_options
        self.output_folder = output_folder
        self.model_name = model_name
        self.validate_petab = validate_petab

    def create_model(
        self,
        force_compile: bool = False,
        verbose: bool = True,
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
        verbose:
            Passed to AMICI's model compilation. If True, the compilation
            progress is printed.
        kwargs:
            Extra arguments passed to amici.SbmlImporter.sbml2amici
        """
        # courtesy check whether target is folder
        if os.path.exists(self.output_folder) and not os.path.isdir(
            self.output_folder
        ):
            raise AssertionError(
                f"Refusing to remove {self.output_folder} for model "
                f"compilation: Not a folder."
            )

        # compile
        if self._must_compile(force_compile):
            logger.info(
                f"Compiling amici model to folder {self.output_folder}."
            )
            if self.petab_problem.model.type_id == MODEL_TYPE_SBML:
                self.compile_model(
                    validate=self.validate_petab,
                    verbose=verbose,
                    **kwargs,
                )
            else:
                self.compile_model(verbose=verbose, **kwargs)
        else:
            logger.debug(
                f"Using existing amici model in folder {self.output_folder}."
            )

        return self._create_model()

    def _create_model(self) -> amici.Model:
        """Load model module and return the model, no checks/compilation."""
        # load moduĺe
        module = amici.import_model_module(
            module_name=self.model_name, module_path=self.output_folder
        )
        model = module.getModel()
        check_model(
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
            amici.import_model_module(self.model_name, self.output_folder)
        except ModuleNotFoundError:
            return True
        except amici.AmiciVersionError as e:
            logger.info(
                f"amici model will be re-imported due to version mismatch: {e}"
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
            Extra arguments passed to :meth:`amici.sbml_import.SbmlImporter.sbml2amici`
            or :func:`amici.pysb_import.pysb2amici`.
        """
        # delete output directory
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        amici.petab.import_petab_problem(
            petab_problem=self.petab_problem,
            model_name=self.model_name,
            model_output_dir=self.output_folder,
            **kwargs,
        )

    def create_solver(
        self,
        model: amici.Model = None,
        verbose: bool = True,
    ) -> amici.Solver:
        """Return model solver."""
        # create model
        if model is None:
            model = self.create_model(verbose=verbose)

        solver = model.getSolver()
        return solver

    def create_edatas(
        self,
        model: amici.Model = None,
        simulation_conditions=None,
        verbose: bool = True,
    ) -> list[amici.ExpData]:
        """Create list of :class:`amici.amici.ExpData` objects."""
        # create model
        if model is None:
            model = self.create_model(verbose=verbose)

        return amici.petab.conditions.create_edatas(
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
        verbose: bool = True,
        **kwargs,
    ) -> AmiciObjective:
        """Create a :class:`pypesto.objective.AmiciObjective`.

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
        verbose:
            Passed to AMICI's model compilation. If True, the compilation
            progress is printed.
        **kwargs:
            Additional arguments passed on to the objective. In case of ordinal
            or semiquantitative measurements, ``inner_options`` can optionally
            be passed here. If none are given, ``inner_options`` given to the
            importer constructor (or inner defaults) will be chosen.

        Returns
        -------
        A :class:`pypesto.objective.AmiciObjective` for the model and the data.
        """
        simulation_conditions = petab.get_simulation_conditions(
            self.petab_problem.measurement_df
        )
        if model is None:
            model = self.create_model(
                force_compile=force_compile, verbose=verbose
            )
        if solver is None:
            solver = self.create_solver(model)
        # create conditions and edatas from measurement data
        if edatas is None:
            edatas = self.create_edatas(
                model=model, simulation_conditions=simulation_conditions
            )
        parameter_mapping = (
            amici.petab.parameter_mapping.create_parameter_mapping(
                petab_problem=self.petab_problem,
                simulation_conditions=simulation_conditions,
                scaled_parameters=True,
                amici_model=model,
                fill_fixed_parameters=False,
            )
        )
        par_ids = self.petab_problem.x_ids

        # fill in dummy parameters (this is needed since some objective
        #  initialization e.g. checks for preeq parameters)
        problem_parameters = dict(
            zip(self.petab_problem.x_ids, self.petab_problem.x_nominal_scaled)
        )
        amici.petab.conditions.fill_in_parameters(
            edatas=edatas,
            problem_parameters=problem_parameters,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=model,
        )

        calculator = None
        amici_reporting = None

        if (
            self._non_quantitative_data_types is not None
            and self._hierarchical
        ):
            inner_options = kwargs.pop("inner_options", None)
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
            amici_reporting = amici.RDataReporting.full

            # FIXME: currently not supported with hierarchical
            if "guess_steadystate" in kwargs and kwargs["guess_steadystate"]:
                warnings.warn(
                    "`guess_steadystate` not supported with hierarchical "
                    "optimization. Disabling `guess_steadystate`.",
                    stacklevel=1,
                )
            kwargs["guess_steadystate"] = False
            inner_parameter_ids = calculator.get_inner_par_ids()
            par_ids = [x for x in par_ids if x not in inner_parameter_ids]

        max_sensi_order = kwargs.get("max_sensi_order", None)

        if (
            self._non_quantitative_data_types is not None
            and any(
                data_type in self._non_quantitative_data_types
                for data_type in [ORDINAL, CENSORED, SEMIQUANTITATIVE]
            )
            and max_sensi_order is not None
            and max_sensi_order > 1
        ):
            raise ValueError(
                "Ordinal, censored and semiquantitative data cannot be "
                "used with second order sensitivities. Use a up to first order "
                "method or disable ordinal, censored and semiquantitative "
            )

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
        post_processor: Callable | None = None,
        post_processor_sensi: Callable | None = None,
        post_processor_time: Callable | None = None,
        max_chunk_size: int | None = None,
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
        A :class:`pypesto.predict.AmiciPredictor` for the model, using
        the outputs of the AMICI model and the timepoints from the PEtab data.
        """
        # if the user didn't pass an objective function, we create it first
        if objective is None:
            objective = self.create_objective()

        # create a identifiers of preequilibration and simulation condition ids
        # which can then be stored in the prediction result
        edata_conditions = objective.amici_object_builder.petab_problem.get_simulation_conditions_from_measurement_df()
        if PREEQUILIBRATION_CONDITION_ID not in list(edata_conditions.columns):
            preeq_dummy = [""] * edata_conditions.shape[0]
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


class PetabSimulatorObjectiveCreator(ObjectiveCreator):
    """ObjectiveCreator for creating an objective based on a PEtabSimulator."""

    def __init__(
        self,
        petab_problem: petab.Problem,
        simulator: Simulator,
    ):
        self.petab_problem = petab_problem
        self.simulator = simulator

    def create_objective(self, **kwargs):
        """Create a PEtabSimulatorObjective."""
        return PetabSimulatorObjective(self.simulator)


class RoadRunnerObjectiveCreator(ObjectiveCreator):
    """ObjectiveCreator for creating an objective for a RoadRunner model."""

    def __init__(
        self,
        petab_problem: petab.Problem,
        rr: roadrunner.RoadRunner | None = None,
    ):
        self.petab_problem = petab_problem
        if rr is None:
            if roadrunner is None:
                raise ImportError(
                    "The `roadrunner` package (on PyPI: `libroadrunner`) "
                    "is required for this objective function."
                )
            rr = roadrunner.RoadRunner()
        self.rr = rr

    def _check_noise_formulae(
        self,
        edatas: list[ExpData] | None = None,
        parameter_mapping: list[ParMappingDictQuadruple] | None = None,
    ):
        """Check if the noise formulae are valid.

        Currently, only static values or singular parameters are supported.
        Complex formulae are not supported.
        """
        # check that parameter mapping is available
        if parameter_mapping is None:
            parameter_mapping = self.create_parameter_mapping()
        # check that edatas are available
        if edatas is None:
            edatas = self.create_edatas()
        # save formulae that need to be changed
        to_change = []
        # check that noise formulae are valid
        for i_edata, (edata, par_map) in enumerate(
            zip(edatas, parameter_mapping)
        ):
            for j_formula, noise_formula in enumerate(edata.noise_formulae):
                # constant values are allowed
                if isinstance(noise_formula, numbers.Number):
                    continue
                # single parameters are allowed
                if noise_formula in par_map[1].keys():
                    continue
                # extract the observable name via regex pattern
                pattern = r"noiseParameter1_(.*?)($|\s)"
                observable_name = re.search(pattern, noise_formula).group(1)
                to_change.append((i_edata, j_formula, observable_name))
        # change formulae
        formulae_changed = []
        for i_edata, j_formula, obs_name in to_change:
            # assign new parameter, formula in RR and parameter into mapping
            original_formula = edatas[i_edata].noise_formulae[j_formula]
            edatas[i_edata].noise_formulae[j_formula] = (
                f"noiseFormula_{obs_name}"
            )
            # different conditions will have the same noise formula
            if (obs_name, original_formula) not in formulae_changed:
                self.rr.addParameter(f"noiseFormula_{obs_name}", 0.0, False)
                self.rr.addAssignmentRule(
                    f"noiseFormula_{obs_name}",
                    original_formula,
                    forceRegenerate=False,
                )
                self.rr.regenerateModel()
                formulae_changed.append((obs_name, original_formula))

    def _write_observables_to_model(self):
        """Write observables of petab problem to the model."""
        # add all observables as species
        for obs_id in self.petab_problem.observable_df.index:
            self.rr.addParameter(obs_id, 0.0, False)
        # extract all parameters from observable formulas
        parameters = petab.get_output_parameters(
            self.petab_problem.observable_df,
            self.petab_problem.model,
            noise=True,
            observables=True,
        )
        # add all parameters to the model
        for param_id in parameters:
            self.rr.addParameter(param_id, 0.0, False)
        formulae = self.petab_problem.observable_df[
            OBSERVABLE_FORMULA
        ].to_dict()

        # add all observable formulas as assignment rules
        for obs_id, formula in formulae.items():
            self.rr.addAssignmentRule(obs_id, formula, forceRegenerate=False)

        # regenerate model to apply changes
        self.rr.regenerateModel()

    def create_edatas(self) -> list[ExpData]:
        """Create a List of :class:`ExpData` objects from the PEtab problem."""
        # Create Dataframes per condition
        return ExpData.from_petab_problem(self.petab_problem)

    def fill_model(self):
        """Fill the RoadRunner model inplace from the PEtab problem.

        Parameters
        ----------
        return_model:
            Flag indicating if the model should be returned.
        """
        if not isinstance(self.petab_problem.model, SbmlModel):
            raise ValueError(
                "The model is not an SBML model. Using "
                "RoadRunner as simulator requires an SBML model."
            )  # TODO: add Pysb support
        if self.petab_problem.model.sbml_document:
            sbml_document = self.petab_problem.model.sbml_document
        elif self.petab_problem.model.sbml_model:
            sbml_document = (
                self.petab_problem.model.sbml_model.getSBMLDocument()
            )
        else:
            raise ValueError("No SBML model found.")
        sbml_writer = libsbml.SBMLWriter()
        sbml_string = sbml_writer.writeSBMLToString(sbml_document)
        self.rr.load(sbml_string)
        self._write_observables_to_model()

    def create_parameter_mapping(self):
        """Create a parameter mapping from the PEtab problem."""
        simulation_conditions = (
            self.petab_problem.get_simulation_conditions_from_measurement_df()
        )
        mapping = petab.get_optimization_to_simulation_parameter_mapping(
            condition_df=self.petab_problem.condition_df,
            measurement_df=self.petab_problem.measurement_df,
            parameter_df=self.petab_problem.parameter_df,
            observable_df=self.petab_problem.observable_df,
            model=self.petab_problem.model,
        )
        # check whether any species in the condition table are assigned
        species = self.rr.model.getFloatingSpeciesIds()
        # overrides in parameter table are handled already
        overrides = [
            specie
            for specie in species
            if specie in self.petab_problem.condition_df.columns
        ]
        if not overrides:
            return mapping
        for (_, condition), mapping_per_condition in zip(
            simulation_conditions.iterrows(), mapping
        ):
            for override in overrides:
                preeq_id = condition.get(PREEQUILIBRATION_CONDITION_ID)
                sim_id = condition.get(SIMULATION_CONDITION_ID)
                if preeq_id:
                    parameter_id_or_value = (
                        self.petab_problem.condition_df.loc[preeq_id, override]
                    )
                    mapping_per_condition[0][override] = parameter_id_or_value
                    if isinstance(parameter_id_or_value, str):
                        mapping_per_condition[2][override] = (
                            self.petab_problem.parameter_df.loc[
                                parameter_id_or_value, petab.PARAMETER_SCALE
                            ]
                        )
                    elif isinstance(parameter_id_or_value, numbers.Number):
                        mapping_per_condition[2][override] = LIN
                    else:
                        raise ValueError(
                            "The parameter value in the condition table "
                            "is not a number or a parameter ID."
                        )
                if sim_id:
                    parameter_id_or_value = (
                        self.petab_problem.condition_df.loc[sim_id, override]
                    )
                    mapping_per_condition[1][override] = parameter_id_or_value
                    if isinstance(parameter_id_or_value, str):
                        mapping_per_condition[3][override] = (
                            self.petab_problem.parameter_df.loc[
                                parameter_id_or_value, petab.PARAMETER_SCALE
                            ]
                        )
                    elif isinstance(parameter_id_or_value, numbers.Number):
                        mapping_per_condition[3][override] = LIN
                    else:
                        raise ValueError(
                            "The parameter value in the condition table "
                            "is not a number or a parameter ID."
                        )
        return mapping

    def create_objective(
        self,
        rr: roadrunner.RoadRunner | None = None,
        edatas: ExpData | None = None,
    ) -> RoadRunnerObjective:
        """Create a :class:`pypesto.objective.RoadRunnerObjective`.

        Parameters
        ----------
        rr:
            RoadRunner instance.
        edatas:
            ExpData object.
        """
        roadrunner_instance = rr
        if roadrunner_instance is None:
            roadrunner_instance = self.rr
            self.fill_model()
        if edatas is None:
            edatas = self.create_edatas()

        parameter_mapping = self.create_parameter_mapping()

        # get x_names
        x_names = self.petab_problem.get_x_ids()

        calculator = RoadRunnerCalculator()

        # run the check for noise formulae
        self._check_noise_formulae(edatas, parameter_mapping)

        return RoadRunnerObjective(
            rr=roadrunner_instance,
            edatas=edatas,
            parameter_mapping=parameter_mapping,
            petab_problem=self.petab_problem,
            calculator=calculator,
            x_names=x_names,
            x_ids=x_names,
        )
