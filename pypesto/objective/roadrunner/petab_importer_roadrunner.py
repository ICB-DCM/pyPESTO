# eventually to be added to pypesto.petab
from __future__ import annotations

import warnings
from typing import Any, Iterable, Optional, Union

import libsbml
import petab
import roadrunner
from petab.models.sbml_model import SbmlModel

from ...petab.importer import PetabStartpoints
from ...problem import Problem
from ...startpoint import StartpointMethod
from ..aggregated import AggregatedObjective
from ..priors import NegLogParameterPriors, get_parameter_prior_dict
from .road_runner import RoadRunnerObjective
from .roadrunner_calculator import RoadRunnerCalculator
from .utils import ExpData


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
        self.petab_problem = petab_problem
        self.rr = roadrunner.RoadRunner()

        self.validate_petab = validate_petab
        if self.validate_petab:
            if petab.lint_problem(petab_problem):
                raise ValueError("Invalid PEtab problem.")

    @staticmethod
    def from_yaml(yaml_config: Union[dict, str]) -> PetabImporterRR:
        """Simplified constructor using a petab yaml file."""
        petab_problem = petab.Problem.from_yaml(yaml_config)

        return PetabImporterRR(petab_problem=petab_problem)

    def _write_observables_to_model(self):
        """Write observables of petab problem to the model."""
        # add all observables as species
        for obs_id in self.petab_problem.observable_df.index:
            try:
                self.rr.addParameter(obs_id, 0.0)
            except RuntimeError:
                warnings.warn(
                    "Observable already exists in model. Skipping.",
                    stacklevel=2,
                )
        # extract all parameters from observable formulas
        parameters = petab.get_output_parameters(
            self.petab_problem.observable_df,
            self.petab_problem.model,
            noise=True,
            observables=True,
        )
        # add all parameters to the model
        for param_id in parameters:
            try:
                self.rr.addParameter(param_id, 0.0)
            except RuntimeError:
                warnings.warn(
                    "Parameter already exists in model. Skipping.",
                    stacklevel=2,
                )
        formulae = self.petab_problem.observable_df[
            "observableFormula"
        ].to_dict()

        # add all observable formulas as assignment rules
        for obs_id, formula in formulae.items():
            if obs_id == list(formulae.keys())[-1]:
                self.rr.addAssignmentRule(obs_id, formula)
                continue
            self.rr.addAssignmentRule(obs_id, formula, forceRegenerate=False)

    def create_edatas(self):
        """Create an ExpData object from the PEtab problem."""
        # Create Dataframes per condition
        grouped_dataframes = {
            key: group
            for key, group in self.petab_problem.measurement_df.groupby(
                "simulationConditionId"
            )
        edatas = [
            ExpData(key, group) for key, group in grouped_dataframes.items()
        ]
        return edatas

    def fill_model(self, return_model: bool = False):
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

        if return_model:
            return self.rr.getModel()

    def create_parameter_mapping(self):
        """Create a parameter mapping from the PEtab problem."""
        return petab.get_optimization_to_simulation_parameter_mapping(
            condition_df=self.petab_problem.condition_df,
            measurement_df=self.petab_problem.measurement_df,
            parameter_df=self.petab_problem.parameter_df,
            observable_df=self.petab_problem.observable_df,
            model=self.petab_problem.model,
        )  # TODO: add sanity checks similar to amici

    def create_objective(
        self,
        rr: Optional[roadrunner.RoadRunner] = None,
        edatas: Optional[ExpData] = None,
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

        return RoadRunnerObjective(
            rr=roadrunner_instance,
            edatas=edatas,
            parameter_mapping=parameter_mapping,
            petab_problem=self.petab_problem,
            calculator=calculator,
            x_names=x_names,
        )

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
        objective: Optional[RoadRunnerObjective] = None,
        x_guesses: Optional[Iterable[float]] = None,
        problem_kwargs: Optional[dict[str, Any]] = None,
        startpoint_kwargs: Optional[dict[str, Any]] = None,
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
            objective = self.create_objective()

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
