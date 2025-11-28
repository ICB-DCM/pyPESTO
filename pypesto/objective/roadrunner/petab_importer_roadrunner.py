"""Importer for PEtab problems using RoadRunner.

Creates from a PEtab problem a roadrunner model, a roadrunner objective or a
pypesto problem with a roadrunner objective. The actual form of the likelihood
depends on the noise model specified in the provided PEtab problem.
"""

from __future__ import annotations

import logging
import numbers
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    import petab.v1 as petab
    from petab.v1.C import (
        OBSERVABLE_FORMULA,
        PREEQUILIBRATION_CONDITION_ID,
        SIMULATION_CONDITION_ID,
    )
    from petab.v1.models.sbml_model import SbmlModel
    from petab.v1.parameter_mapping import ParMappingDictQuadruple
except ImportError:
    petab = None

import pypesto.C

from ...problem import Problem
from ...startpoint import StartpointMethod
from ..aggregated import AggregatedObjective
from ..priors import NegLogParameterPriors, get_parameter_prior_dict
from .road_runner import RoadRunnerObjective
from .roadrunner_calculator import RoadRunnerCalculator
from .utils import ExpData

try:
    import libsbml
    import roadrunner
except ImportError:
    roadrunner = None
    libsbml = None

logger = logging.getLogger(__name__)


class PetabImporterRR:
    """
    Importer for PEtab problems using RoadRunner.

    Create a :class:`roadrunner.RoadRunner` instance,
    a :class:`pypesto.objective.RoadRunnerObjective` or a
    :class:`pypesto.problem.Problem` from PEtab files. The actual
    form of the likelihood depends on the noise model specified in the provided PEtab problem.
    For more information, see the
    `PEtab documentation <https://petab.readthedocs.io/en/latest/documentation_data_format.html#noise-distributions>`_.
    """  # noqa

    def __init__(
        self, petab_problem: petab.Problem, validate_petab: bool = True
    ):
        """Initialize importer.

        Parameters
        ----------
        petab_problem:
            Managing access to the model and data.
        validate_petab:
            Flag indicating if the PEtab problem shall be validated.
        """
        warnings.warn(
            "The RoadRunner importer is deprecated and will be removed in "
            "future versions. Please use the generic PetabImporter instead "
            "with `simulator_type='roadrunner'`. Everything else will stay "
            "same.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.petab_problem = petab_problem
        if validate_petab:
            if petab.lint_problem(petab_problem):
                raise ValueError("Invalid PEtab problem.")
        self.rr = roadrunner.RoadRunner()

    @staticmethod
    def from_yaml(yaml_config: Path | str) -> PetabImporterRR:
        """Simplified constructor using a petab yaml file."""
        petab_problem = petab.Problem.from_yaml(yaml_config)

        return PetabImporterRR(petab_problem=petab_problem)

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
                        mapping_per_condition[2][override] = pypesto.C.LIN
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
                        mapping_per_condition[3][override] = pypesto.C.LIN
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

    def create_prior(self) -> NegLogParameterPriors | None:
        """
        Create a prior from the parameter table.

        Returns None, if no priors are defined.
        """
        prior_list = []

        if petab.OBJECTIVE_PRIOR_TYPE not in self.petab_problem.parameter_df:
            return None

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
        return NegLogParameterPriors(prior_list)

    def create_startpoint_method(self, **kwargs) -> StartpointMethod:
        """Create a startpoint method.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments passed on to
            :meth:`pypesto.startpoint.FunctionStartpoints.__init__`.
        """
        from ...petab.util import PetabStartpoints

        return PetabStartpoints(petab_problem=self.petab_problem, **kwargs)

    def create_problem(
        self,
        objective: RoadRunnerObjective | None = None,
        x_guesses: Iterable[float] | None = None,
        problem_kwargs: dict[str, Any] | None = None,
        startpoint_kwargs: dict[str, Any] | None = None,
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
        A :class:`pypesto.problem.Problem` instance.
        """
        if objective is None:
            objective = self.create_objective(**kwargs)

        x_fixed_indices = self.petab_problem.x_fixed_indices
        x_fixed_vals = self.petab_problem.x_nominal_fixed_scaled
        x_ids = self.petab_problem.x_ids
        lb = self.petab_problem.lb_scaled
        ub = self.petab_problem.ub_scaled

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
                **startpoint_kwargs
            ),
            copy_objective=False,
            **problem_kwargs,
        )

        return problem
