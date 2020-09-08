import pandas as pd
import os
import sys
import importlib
import shutil
import logging
import tempfile
from typing import List, Sequence, Union

from ..problem import Problem
from ..objective import AmiciObjective, AmiciObjectBuilder, AggregatedObjective
from ..objective.priors import NegLogParameterPriors, \
    get_parameter_prior_dict

try:
    import petab
    import amici
    import amici.petab_import
    import amici.petab_objective
    import amici.parameter_mapping
except ImportError:
    pass

logger = logging.getLogger(__name__)


class PetabImporter(AmiciObjectBuilder):
    MODEL_BASE_DIR = "amici_models"

    def __init__(self,
                 petab_problem: 'petab.Problem',
                 output_folder: str = None,
                 model_name: str = None):
        """
        petab_problem:
            Managing access to the model and data.
        output_folder:
            Folder to contain the amici model. Defaults to
            './amici_models/{model_name}'.
        model_name:
            Name of the model, which will in particular be the name of the
            compiled model python module.
        """
        self.petab_problem = petab_problem

        if output_folder is None:
            output_folder = _find_output_folder_name(self.petab_problem)
        self.output_folder = output_folder

        if model_name is None:
            model_name = _find_model_name(self.output_folder)
        self.model_name = model_name

    @staticmethod
    def from_yaml(yaml_config: Union[dict, str],
                  output_folder: str = None,
                  model_name: str = None) -> 'PetabImporter':
        """
        Simplified constructor using a petab yaml file.
        """
        petab_problem = petab.Problem.from_yaml(yaml_config)

        return PetabImporter(
            petab_problem=petab_problem,
            output_folder=output_folder,
            model_name=model_name)

    def create_model(self,
                     force_compile: bool = False,
                     **kwargs) -> 'amici.Model':
        """
        Import amici model. If necessary or force_compile is True, compile
        first.

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
        if os.path.exists(self.output_folder) \
                and not os.path.isdir(self.output_folder):
            raise AssertionError(
                f"Refusing to remove {self.output_folder} for model "
                f"compilation: Not a folder.")

        # add module to path
        if self.output_folder not in sys.path:
            sys.path.insert(0, self.output_folder)

        # compile
        if self._must_compile(force_compile):
            logger.info(f"Compiling amici model to folder "
                        f"{self.output_folder}.")
            self.compile_model(**kwargs)
        else:
            logger.info(f"Using existing amici model in folder "
                        f"{self.output_folder}.")

        return self._create_model()

    def _create_model(self) -> 'amici.Model':
        """
        No checks, no compilation, just load the model module and return
        the model.
        """
        # load moduĺe
        module = amici.import_model_module(module_name=self.model_name,
                                           module_path=self.output_folder)
        model = module.getModel()

        return model

    def _must_compile(self, force_compile: bool):
        """
        Check whether the model needs to be compiled first.
        """
        # asked by user
        if force_compile:
            return True

        # folder does not exist
        if not os.path.exists(self.output_folder) or \
                not os.listdir(self.output_folder):
            return True

        # try to import (in particular checks version)
        try:
            # importing will already raise an exception if version wrong
            importlib.import_module(self.model_name)
        except ModuleNotFoundError:
            return True

        # no need to (re-)compile
        return False

    def compile_model(self, **kwargs):
        """
        Compile the model. If the output folder exists already, it is first
        deleted.

        Parameters
        ----------
        kwargs: Extra arguments passed to `amici.SbmlImporter.sbml2amici`.

        """

        # delete output directory
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        amici.petab_import.import_model(
            sbml_model=self.petab_problem.sbml_model,
            condition_table=self.petab_problem.condition_df,
            observable_table=self.petab_problem.observable_df,
            model_name=self.model_name,
            model_output_dir=self.output_folder,
            **kwargs)

    def create_solver(self, model: 'amici.Model' = None) -> 'amici.Solver':
        """
        Return model solver.
        """
        # create model
        if model is None:
            model = self.create_model()

        solver = model.getSolver()
        return solver

    def create_edatas(
            self,
            model: 'amici.Model' = None,
            simulation_conditions=None
    ) -> List['amici.ExpData']:
        """
        Create list of amici.ExpData objects.
        """
        # create model
        if model is None:
            model = self.create_model()

        return amici.petab_objective.create_edatas(
            amici_model=model,
            petab_problem=self.petab_problem,
            simulation_conditions=simulation_conditions)

    def create_objective(
            self,
            model: 'amici.Model' = None,
            solver: 'amici.Solver' = None,
            edatas: Sequence['amici.ExpData'] = None,
            force_compile: bool = False,
            **kwargs
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
            Additional arguments passed on to the objective.

        Returns
        -------
        objective:
            A :class:`pypesto.AmiciObjective` for the model and the data.
        """
        # get simulation conditions
        simulation_conditions = petab.get_simulation_conditions(
            self.petab_problem.measurement_df)

        # create model
        if model is None:
            model = self.create_model(force_compile=force_compile)
        # create solver
        if solver is None:
            solver = self.create_solver(model)
        # create conditions and edatas from measurement data
        if edatas is None:
            edatas = self.create_edatas(
                model=model,
                simulation_conditions=simulation_conditions)

        parameter_mapping = amici.petab_objective.create_parameter_mapping(
            petab_problem=self.petab_problem,
            simulation_conditions=simulation_conditions,
            scaled_parameters=True,
            amici_model=model)

        par_ids = self.petab_problem.x_ids

        # fill in dummy parameters (this is needed since some objective
        #  initialization e.g. checks for preeq parameters)
        problem_parameters = {key: val for key, val in zip(
            self.petab_problem.x_ids,
            self.petab_problem.x_nominal_scaled)}
        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=problem_parameters,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=model)

        # create objective
        obj = AmiciObjective(
            amici_model=model, amici_solver=solver, edatas=edatas,
            x_ids=par_ids, x_names=par_ids,
            parameter_mapping=parameter_mapping,
            amici_object_builder=self,
            **kwargs)

        return obj

    def create_prior(self) -> NegLogParameterPriors:
        """
        Creates a prior from the parameter table. Returns None, if no priors
        are defined.
        """

        prior_list = []

        if petab.OBJECTIVE_PRIOR_TYPE in self.petab_problem.parameter_df:

            for i, x_id in enumerate(self.petab_problem.x_ids):

                prior_type_entry = self.petab_problem.\
                    parameter_df.loc[x_id, petab.OBJECTIVE_PRIOR_TYPE]

                if (isinstance(prior_type_entry, str)
                        and prior_type_entry != petab.PARAMETER_SCALE_UNIFORM):

                    prior_params = [float(param) for param in
                                    self.petab_problem.parameter_df.
                                    loc[x_id, petab.OBJECTIVE_PRIOR_PARAMETERS]
                                    .split(';')]

                    scale = self.petab_problem.\
                        parameter_df.loc[x_id, petab.PARAMETER_SCALE]

                    prior_list.append(
                        get_parameter_prior_dict(i,
                                                 prior_type_entry,
                                                 prior_params,
                                                 scale))

        if len(prior_list):
            return NegLogParameterPriors(prior_list)
        else:
            return None

    def create_problem(
            self, objective: AmiciObjective = None, **kwargs
    ) -> Problem:
        """Create a :class:`pypesto.Problem`.

        Parameters
        ----------
        objective:
            Objective as created by `create_objective`.
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

        prior = self.create_prior()

        if prior is not None:
            objective = AggregatedObjective([objective, prior])

        x_scales = \
            [self.petab_problem.parameter_df.loc[x_id, petab.PARAMETER_SCALE]
                for x_id in self.petab_problem.x_ids]

        problem = Problem(
            objective=objective,
            lb=self.petab_problem.lb_scaled,
            ub=self.petab_problem.ub_scaled,
            x_fixed_indices=self.petab_problem.x_fixed_indices,
            x_fixed_vals=self.petab_problem.x_nominal_fixed_scaled,
            x_names=self.petab_problem.x_ids,
            x_scales=x_scales,
            x_priors_defs=prior)

        return problem

    def rdatas_to_measurement_df(
            self, rdatas: Sequence['amici.ReturnData'],
            model: 'amici.Model' = None
    ) -> pd.DataFrame:
        """
        Create a measurement dataframe in the petab format from
        the passed `rdatas` and own information.

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
            rdatas, model, measurement_df)

    def rdatas_to_simulation_df(
            self, rdatas: Sequence['amici.ReturnData'],
            model: 'amici.Model' = None
    ) -> pd.DataFrame:
        """Same as `rdatas_to_measurement_df`, execpt a petab simulation
        dataframe is created, i.e. the measurement column label is adjusted.
        """
        return self.rdatas_to_measurement_df(rdatas, model).rename(
            {petab.MEASUREMENT: petab.SIMULATION})


def _find_output_folder_name(petab_problem: 'petab.Problem') -> str:
    """
    Find a name for storing the compiled amici model in. If available,
    use the sbml model name from the `petab_problem`, otherwise create
    a unique name.
    The folder will be located in the `PetabImporter.MODEL_BASE_DIR`
    subdirectory of the current directory.
    """
    # check whether location for amici model is a file
    if os.path.exists(PetabImporter.MODEL_BASE_DIR) and \
            not os.path.isdir(PetabImporter.MODEL_BASE_DIR):
        raise AssertionError(
            f"{PetabImporter.MODEL_BASE_DIR} exists and is not a directory, "
            f"thus cannot create a directory for the compiled amici model.")

    # create base directory if non-existent
    if not os.path.exists(PetabImporter.MODEL_BASE_DIR):
        os.makedirs(PetabImporter.MODEL_BASE_DIR)

    # try sbml model id
    sbml_model_id = petab_problem.sbml_model.getId()
    if sbml_model_id:
        output_folder = os.path.abspath(
            os.path.join(PetabImporter.MODEL_BASE_DIR, sbml_model_id))
    else:
        # create random folder name
        output_folder = os.path.abspath(
            tempfile.mkdtemp(dir=PetabImporter.MODEL_BASE_DIR))
    return output_folder


def _find_model_name(output_folder: str) -> str:
    """
    Just re-use the last part of the output folder.
    """
    return os.path.split(os.path.normpath(output_folder))[-1]
