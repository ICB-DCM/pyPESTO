import numpy as np
import pandas as pd
import os
import sys
import importlib
import copy
import shutil
import re
import logging
import tempfile

from typing import List

import petab
from petab.C import NOMINAL_VALUE
from ..problem import Problem
from .amici_objective import AmiciObjective
from amici.petab_import import import_model
from amici.petab_objective import edatas_from_petab, rdatas_to_measurement_df

try:
    import amici
    import amici.petab_import
except ImportError:
    amici = None


logger = logging.getLogger(__name__)


class PetabImporter:

    MODEL_BASE_DIR = "amici_models"

    def __init__(self,
                 petab_problem: petab.Problem,
                 output_folder: str = None,
                 model_name: str = None):
        """
        petab_problem: petab.Problem
            Managing access to the model and data.

        output_folder: str,  optional
            Folder to contain the amici model. Defaults to
            './amici_models/model_name'.

        model_name: str, optional
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
    def from_folder(folder,
                    output_folder: str = None,
                    model_name: str = None):
        """
        Simplified constructor exploiting the standardized petab folder
        structure.

        Parameters
        ----------

        folder: str
            Path to the base folder of the model, as in
            petab.Problem.from_folder.
        output_folder: See __init__.
        model_name: See __init__.
        """
        petab_problem = petab.Problem.from_folder(folder)

        return PetabImporter(
            petab_problem=petab_problem,
            output_folder=output_folder,
            model_name=model_name)

    def create_model(self, force_compile=False,
                     *args, **kwargs) -> amici.Model:
        """
        Import amici model. If necessary or force_compile is True, compile
        first.

        Parameters
        ----------

        force_compile: str, optional
            If False, the model is compiled only if the output folder does not
            exist yet. If True, the output folder is deleted and the model
            (re-)compiled in either case.

            .. warning::
                If `force_compile`, then an existing folder of that name will
                be deleted.

        args, kwargs: Extra arguments passed to amici.SbmlImporter.sbml2amici
        """
        # courtesy check if target not folder
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
            self.compile_model(*args, **kwargs)
        else:
            logger.info(f"Using existing amici model in folder "
                        f"{self.output_folder}.")

        return self._create_model()

    def _create_model(self) -> amici.Model:
        """
        No checks, no compilation, just load the model module and return
        the model.
        """
        # load moduĺe
        model_module = importlib.import_module(self.model_name)

        # import model
        model = model_module.getModel()

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
        except RuntimeError:
            return True

        # no need to (re-)compile
        return False

    def compile_model(self, **kwargs):
        """
        Compile the model. If the output folder exists already, it is first
        deleted.

        Parameters
        ----------
        kwargs: Extra arguments passed to amici.SbmlImporter.sbml2amici

        """

        # delete output directory
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        import_model(sbml_model=self.petab_problem.sbml_model,
                     condition_table=self.petab_problem.condition_df,
                     observable_table=self.petab_problem.observable_df,
                     model_name=self.model_name,
                     model_output_dir=self.output_folder,
                     **kwargs)

    def create_solver(self, model=None):
        """
        Return model solver.
        """
        # create model
        if model is None:
            model = self.create_model()

        solver = model.getSolver()
        return solver

    def create_edatas(self, model: amici.Model = None,
                      simulation_conditions=None) -> List[amici.ExpData]:
        """
        Create list of amici.ExpData objects.
        """
        # create model
        if model is None:
            model = self.create_model()

        problem_parameters = {t.Index: getattr(t, NOMINAL_VALUE) for t in
                              self.petab_problem.parameter_df.itertuples()}

        return edatas_from_petab(
            model=model,
            petab_problem=self.petab_problem,
            problem_parameters=problem_parameters,
            simulation_conditions=simulation_conditions)

    def create_objective(self,
                         model=None,
                         solver=None,
                         edatas=None,
                         force_compile: bool = False):
        """
        Create a pypesto.PetabAmiciObjective.
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

        # simulation <-> optimization parameter mapping
        par_opt_ids = self.petab_problem.get_optimization_parameters()

        parameter_mappings = \
            petab.get_optimization_to_simulation_parameter_mapping(
                condition_df=self.petab_problem.condition_df,
                measurement_df=self.petab_problem.measurement_df,
                parameter_df=self.petab_problem.parameter_df,
                sbml_model=self.petab_problem.sbml_model,
                simulation_conditions=simulation_conditions,
            )

        scale_mappings = \
            petab.get_optimization_to_simulation_scale_mapping(
                parameter_df=self.petab_problem.parameter_df,
                mapping_par_opt_to_par_sim=parameter_mappings,
                measurement_df=self.petab_problem.measurement_df
            )

        # unify and check preeq and sim mappings
        parameter_mapping, scale_mapping = petab.merge_preeq_and_sim_pars(
            parameter_mappings, scale_mappings)

        # simulation ids (for correct order)
        par_sim_ids = list(model.getParameterIds())

        # create lists from dicts in correct order
        parameter_mapping = _mapping_to_list(parameter_mapping, par_sim_ids)
        scale_mapping = _mapping_to_list(scale_mapping, par_sim_ids)

        # create objective
        obj = PetabAmiciObjective(
            petab_importer=self,
            amici_model=model, amici_solver=solver, edatas=edatas,
            x_ids=par_opt_ids, x_names=par_opt_ids,
            mapping_par_opt_to_par_sim=parameter_mapping,
            mapping_scale_opt_to_scale_sim=scale_mapping
        )

        return obj

    def create_problem(self, objective):
        problem = Problem(objective=objective,
                          lb=self.petab_problem.lb_scaled,
                          ub=self.petab_problem.ub_scaled,
                          x_fixed_indices=self.petab_problem.x_fixed_indices,
                          x_fixed_vals=self.petab_problem.x_fixed_vals_scaled,
                          x_names=self.petab_problem.x_ids)

        return problem

    def rdatas_to_measurement_df(self, rdatas, model=None) -> pd.DataFrame:
        """
        Create a measurement dataframe in the petab format from
        the passed `rdatas` and own information.

        Parameters
        ----------

        rdatas: list of amici.RData
            A list of rdatas as produced by
            pypesto.AmiciObjective.__call__(x, return_dict=True)['rdatas'].

        Returns
        -------

        df: pandas.DataFrame
            A dataframe built from the rdatas in the format as in
            self.petab_problem.measurement_df.
        """
        # create model
        if model is None:
            model = self.create_model()

        measurement_df = self.petab_problem.measurement_df

        return rdatas_to_measurement_df(rdatas, model, measurement_df)


def _find_output_folder_name(petab_problem: petab.Problem):
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


def _find_model_name(output_folder):
    """
    Just re-use the last part of the output folder.
    """
    return os.path.split(os.path.normpath(output_folder))[-1]


def _mapping_to_list(mapping, par_sim_ids):
    """
    Petab returns for each condition a dictionary which maps simulation
    to optimization parameters. Given we know the correct order of
    simulation parameters as used in the amici model, we here create
    a list from the dictionary.

    Parameters
    ----------
    mapping: list of dict
        as created by _merge_preeq_and_sim_pars.
    par_sim_ids: list of str
        The simulation ids as returned by list(amici_model.getParameterIds()).

    Returns
    -------
    mapping_list: list of list
        Each dict turned into a list with order according to `par_sim_ids`.
    """
    mapping_list = []
    for map_for_cond in mapping:
        map_for_cond_list = []
        for sim_id in par_sim_ids:
            map_for_cond_list.append(map_for_cond[sim_id])
        mapping_list.append(map_for_cond_list)
    return mapping_list


class PetabAmiciObjective(AmiciObjective):
    """
    This is a shallow wrapper around AmiciObjective to make it serializable.
    """

    def __init__(
            self,
            petab_importer,
            amici_model, amici_solver, edatas,
            x_ids, x_names,
            mapping_par_opt_to_par_sim,
            mapping_scale_opt_to_scale_sim):

        super().__init__(
            amici_model=amici_model,
            amici_solver=amici_solver,
            edatas=edatas,
            x_ids=x_ids, x_names=x_names,
            mapping_par_opt_to_par_sim=mapping_par_opt_to_par_sim,
            mapping_scale_opt_to_scale_sim=mapping_scale_opt_to_scale_sim)

        self.petab_importer = petab_importer

    def __getstate__(self):
        state = {}
        for key in set(self.__dict__.keys()) - \
                set(['amici_model', 'amici_solver', 'edatas']):
            state[key] = self.__dict__[key]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        petab_importer = state['petab_importer']

        model = petab_importer.create_model()
        solver = petab_importer.create_solver(model)
        edatas = petab_importer.create_edatas(model)

        self.amici_model = model
        self.amici_solver = solver
        self.edatas = edatas

    def __deepcopy__(self, memodict=None):
        other = self.__class__.__new__(self.__class__)

        for key in set(self.__dict__.keys()) - \
                set(['amici_model', 'amici_solver', 'edatas']):
            other.__dict__[key] = copy.deepcopy(self.__dict__[key])

        other.amici_model = amici.ModelPtr(self.amici_model.clone())
        other.amici_solver = amici.SolverPtr(self.amici_solver.clone())
        other.edatas = [amici.ExpData(data) for data in self.edatas]

        return other
