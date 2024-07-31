"""Contains the PetabImporter class."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import warnings
from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Callable,
)

import petab.v1 as petab
from petab.v1.C import (
    PREEQUILIBRATION_CONDITION_ID,
    SIMULATION_CONDITION_ID,
)
from petab.v1.models import MODEL_TYPE_SBML

from ..C import (
    CENSORED,
    CONDITION_SEP,
    ORDINAL,
    SEMIQUANTITATIVE,
)
from ..hierarchical.inner_calculator_collector import InnerCalculatorCollector
from ..objective import AmiciObjective
from ..objective.amici import AmiciObjectBuilder
from ..predict import AmiciPredictor

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


class AmiciFactory(AmiciObjectBuilder):
    """Factory for creating an amici objective function."""

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

        # add module to path
        if self.output_folder not in sys.path:
            sys.path.insert(0, self.output_folder)

        # compile
        if self._must_compile(force_compile):
            logger.info(
                f"Compiling amici model to folder " f"{self.output_folder}."
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
